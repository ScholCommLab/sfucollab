"""
SIAT research dashboard.

Reads siat_data/works.json and siat_data/roster.json (written by the collection
notebook) and serves an interactive dashboard in the browser.

    pip install dash dash-cytoscape networkx pandas plotly
    python app.py            # then open http://127.0.0.1:8050

Environment variables:
    SIAT_DATA   folder holding works.json and roster.json (default: siat_data)
    PORT        port to serve on (default: 8050)

Updating: re-run the collection notebook. The app notices the new files on the
next interaction (no restart needed). Refresh the page to pick up roster changes
in the "focus" dropdowns.
"""
import itertools
import json
import math
import os
from collections import Counter, defaultdict
from datetime import date
from functools import lru_cache
from pathlib import Path

import dash_cytoscape as cyto
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

try:
    import pycountry
except ImportError:
    pycountry = None

DATA_DIR = Path(os.environ.get("SIAT_DATA", "siat_data"))
WORKS_FILE = DATA_DIR / "works.json"
ROSTER_FILE = DATA_DIR / "roster.json"
RED, BLUE, GREY = "#c8102e", "#1f6feb", "#9aa5b1"
GROUP_COLORS = {"Research faculty": RED, "Teaching faculty": BLUE}
DEFAULT_GROUP = "Research faculty"
DEFAULT_YEARS = 5


# --------------------------------------------------------------------------- data
def _stamp():
    g = DATA_DIR / "groups.json"
    return (WORKS_FILE.stat().st_mtime, ROSTER_FILE.stat().st_mtime, g.stat().st_mtime if g.exists() else 0)


@lru_cache(maxsize=1)
def _load(stamp):
    blob = json.loads(WORKS_FILE.read_text())
    roster = json.loads(ROSTER_FILE.read_text())
    groups_file = DATA_DIR / "groups.json"
    groups = json.loads(groups_file.read_text()) if groups_file.exists() else {}
    groups = {n: groups.get(n, DEFAULT_GROUP) for n in roster}
    full = lambda i: i if i.startswith("http") else "https://openalex.org/" + i
    id2m = {full(i): name for name, ids in roster.items() for i in ids}

    labels, keys, rows = {}, {}, []
    ext, topic, inst_info = {}, {}, {}
    for wid, w in blob["works"].items():
        ks, members = [], []
        for au in w.get("authorships") or []:
            a = au.get("author") or {}
            aid = a.get("id")
            if not aid:
                continue
            if aid in id2m:
                k = id2m[aid]
                labels[k] = k
                if k not in members:
                    members.append(k)
            else:
                k = aid.rsplit("/", 1)[-1]
                labels[k] = a.get("display_name") or k
                insts = []
                for i in au.get("institutions") or []:
                    if i.get("id"):
                        inst_info[i["id"]] = (i.get("display_name") or i["id"], i.get("country_code") or "", i.get("type") or "")
                        insts.append(i["id"])
                ccs = list(au.get("countries") or []) or sorted({inst_info[i][1] for i in insts if inst_info[i][1]})
                ext.setdefault(wid, {})[k] = (tuple(insts), tuple(ccs))
            if k not in ks:
                ks.append(k)
        keys[wid] = ks
        pt = w.get("primary_topic") or {}
        if pt.get("id"):
            nm = lambda x: (pt.get(x) or {}).get("display_name") or "Unknown"
            topic[wid] = (nm("domain"), nm("field"), nm("subfield"), pt.get("display_name") or "Unknown")
        if w.get("publication_year") is None:
            continue
        rows.append(dict(id=wid, title=w.get("title"), year=int(w["publication_year"]),
                         type=w.get("type") or "unknown", cited=w.get("cited_by_count") or 0,
                         members=members))
    df = pd.DataFrame(rows, columns=["id", "title", "year", "type", "cited", "members"])
    return dict(roster=set(roster), groups=groups, ext=ext, topic=topic, inst=inst_info, labels=labels, keys=keys, works=blob["works"], df=df,
                fetched=blob.get("fetched", "unknown"))


def corpus():
    return _load(_stamp())


def window(n_years, end_year):
    return end_year - n_years + 1, end_year


def work_ids_in(c, start, end):
    df = c["df"]
    return df.loc[df.year.between(start, end), "id"].tolist()


def h_index(cites):
    cites = sorted(cites, reverse=True)
    return sum(1 for i, x in enumerate(cites, 1) if x >= i)


# ------------------------------------------------------------------------- graphs
@lru_cache(maxsize=32)
def co_graph(stamp, start, end, max_authors):
    """Undirected co-authorship graph. Edge weight = number of joint works."""
    c = corpus()
    G, cnt = nx.Graph(), Counter()
    for wid in work_ids_in(c, start, end):
        ks = c["keys"][wid]
        if len(ks) > max_authors:
            continue
        cnt.update(ks)
        for u, v in itertools.combinations(sorted(ks), 2):
            if G.has_edge(u, v):
                G[u][v]["weight"] += 1
            else:
                G.add_edge(u, v, weight=1)
    for n, k in cnt.items():
        if n not in G:
            G.add_node(n)
        G.nodes[n]["value"] = k
    return G


@lru_cache(maxsize=32)
def ci_graph(stamp, start, end, max_authors):
    """Directed graph: citing author -> cited author, within the collected corpus.
    Citing works are limited to the window; cited works can be of any year."""
    c = corpus()
    keys = c["keys"]
    G = nx.DiGraph()
    for wid in work_ids_in(c, start, end):
        citing = keys[wid]
        if len(citing) > max_authors:
            continue
        for ref in c["works"][wid].get("referenced_works") or []:
            cited = keys.get(ref)
            if not cited or len(cited) > max_authors:
                continue
            for u in citing:
                for v in cited:
                    if u == v:
                        continue
                    if G.has_edge(u, v):
                        G[u][v]["weight"] += 1
                    else:
                        G.add_edge(u, v, weight=1)
    for n in G:
        G.nodes[n]["value"] = G.in_degree(n, weight="weight")
    return G


def prune(G, roster, include_ext, ext_ext, min_ties, top_n, min_edge, focus):
    """Shrink a graph to something readable. Order matters: edge filters, ego, then externals."""
    H = G.copy()
    H.remove_edges_from([
        (u, v) for u, v, d in H.edges(data=True)
        if d["weight"] < min_edge or (not ext_ext and u not in roster and v not in roster)
    ])
    if focus:
        if focus not in H:
            return H.__class__()
        nb = set(H.successors(focus)) | set(H.predecessors(focus)) if H.is_directed() else set(H[focus])
        H = H.subgraph(nb | {focus}).copy()
    if not include_ext:
        H.remove_nodes_from([n for n in H if n not in roster])
    else:
        ties = Counter()  # weight of links from each external author to SIAT members
        for u, v, d in H.edges(data=True):
            if u in roster and v not in roster:
                ties[v] += d["weight"]
            elif v in roster and u not in roster:
                ties[u] += d["weight"]
        keep = {n for n, t in ties.most_common(top_n) if t >= min_ties}
        H.remove_nodes_from([n for n in H if n not in roster and n not in keep])
    return H


def to_elements(H, roster, value_label, equal=False):
    labels, groups = corpus()["labels"], corpus()["groups"]
    nb = defaultdict(Counter)
    for u, v, d in H.edges(data=True):
        nb[u][v] += d["weight"]
        nb[v][u] += d["weight"]
    vmax = max([d.get("value", 0) for _, d in H.nodes(data=True)] or [1]) or 1
    els = []
    for n, d in H.nodes(data=True):
        siat = n in roster
        val = d.get("value", 0)
        if equal:
            size = 30 if siat else 12
        else:  # area grows with value, relative to the biggest node shown; SIAT members never shrink below 16
            size = 6 + 54 * math.sqrt(val / vmax)
            if siat:
                size = max(size, 16)
        top = ", ".join(f"{labels[m]} ({w})" for m, w in nb[n].most_common(6))
        els.append({"data": {"id": n, "label": labels[n], "size": round(size, 1), "value": val,
                             "value_label": value_label,
                             "kind": groups[n] if siat else "External author",
                             "color": GROUP_COLORS.get(groups[n], RED) if siat else GREY, "nbrs": top},
                    "classes": "siat" if siat else "ext"})
    for u, v, d in H.edges(data=True):
        els.append({"data": {"source": u, "target": v, "weight": d["weight"]},
                    "classes": "core" if (u in roster and v in roster) else "periph"})
    return els


def stylesheet(directed, max_w, label_ext):
    ext = {"background-color": GREY}
    if label_ext:
        ext.update({"label": "data(label)", "font-size": 8, "color": "#555",
                    "text-valign": "bottom", "text-margin-y": 3})
    edge = {"width": f"mapData(weight, 1, {max(2, max_w)}, 1, 8)", "curve-style": "bezier", "opacity": 0.7}
    if directed:
        edge.update({"target-arrow-shape": "triangle", "arrow-scale": 0.8})
    return [
        {"selector": "node", "style": {"width": "data(size)", "height": "data(size)", "background-color": GREY}},
        {"selector": "node.ext", "style": ext},
        {"selector": "node.siat", "style": {"background-color": "data(color)", "label": "data(label)", "font-size": 11,
                                            "font-weight": "bold", "text-valign": "bottom",
                                            "text-margin-y": 4, "z-index": 10}},
        {"selector": "edge", "style": edge},
        {"selector": "edge.periph", "style": {"line-color": "#d3d8de", "target-arrow-color": "#d3d8de"}},
        {"selector": "edge.core", "style": {"line-color": "#8a8f98", "target-arrow-color": "#8a8f98"}},
        {"selector": ":selected", "style": {"border-width": 3, "border-color": "#111"}},
    ]


LAYOUTS = {
    "cose": {"name": "cose", "animate": False, "randomize": True, "nodeRepulsion": 400000,
             "idealEdgeLength": 100, "gravity": 80, "numIter": 1000},
    "concentric": {"name": "concentric", "animate": False, "minNodeSpacing": 20},
    "circle": {"name": "circle", "animate": False},
    "grid": {"name": "grid", "animate": False},
}


def graph_view(directed, n_years, end_year, opts, min_ties, top_n, min_edge, max_auth, focus, layout_name,
               size_mode="scaled"):
    c = corpus()
    start, end = window(n_years, end_year)
    build = ci_graph if directed else co_graph
    G = build(_stamp(), start, end, max_auth)
    H = prune(G, c["roster"], "ext" in opts, "extext" in opts, min_ties, top_n, min_edge, focus)
    els = to_elements(H, c["roster"], "Citations received within the corpus" if directed else "Works in window",
                      equal=size_mode == "equal")
    max_w = max([d["weight"] for _, _, d in H.edges(data=True)] or [1])
    n_siat = sum(1 for n in H if n in c["roster"])
    stats = (f"{H.number_of_nodes()} nodes ({n_siat} SIAT, {H.number_of_nodes() - n_siat} external), "
             f"{H.number_of_edges()} edges, {start}-{end}. Click a node for details.")
    return els, stylesheet(directed, max_w, "labels" in opts), LAYOUTS[layout_name], stats


# ----------------------------------------------------------------------- overview
def overview_view(n_years, end_year):
    c = corpus()
    start, end = window(n_years, end_year)
    df = c["df"]
    d = df[df.year.between(start, end)]
    per_member = defaultdict(list)
    for r in d.itertuples():
        for m in r.members:
            per_member[m].append(r.cited)
    ext_authors = {k for wid in d.id for k in c["keys"][wid] if k not in c["roster"]}

    years = list(range(start, end + 1))
    outs = d.groupby("year").size().reindex(years, fill_value=0)
    cites = d.groupby("year")["cited"].sum().reindex(years, fill_value=0)
    types = d.groupby("type").size().sort_values()

    def bar(x, y, title, horizontal=False):
        data = pd.DataFrame({"label": list(x), "n": list(y)})
        fig = px.bar(data, x="n" if horizontal else "label", y="label" if horizontal else "n",
                     title=title, orientation="h" if horizontal else "v")
        fig.update_traces(marker_color=RED)
        fig.update_layout(height=300, margin=dict(t=40, b=30, l=40, r=10),
                          xaxis_title=None, yaxis_title=None, xaxis_type="category" if not horizontal else None)
        return fig

    f_out = bar([str(y) for y in years], outs.values, "Outputs per year")
    f_cit = bar([str(y) for y in years], cites.values, "Citations to date, by publication year")
    f_typ = bar(list(types.index), types.values, "Outputs by type", horizontal=True)

    table = pd.DataFrame([dict(author=m, group=c["groups"][m], outputs=len(cs), citations=sum(cs), h_index=h_index(cs))
                          for m, cs in per_member.items()],
                         columns=["author", "group", "outputs", "citations", "h_index"])
    table = table.sort_values("citations", ascending=False)

    def card(label, value):
        return html.Div([html.Div(f"{value:,}", style={"fontSize": "28px", "fontWeight": 600}),
                         html.Div(label, style={"color": "#666", "fontSize": "13px"})], style=CARD)

    kpis = [card("Outputs (unique works)", len(d)), card("Citations to date", int(d.cited.sum())),
            card("SIAT members with outputs", len(per_member)), card("External co-authors", len(ext_authors))]
    label = f"Showing {start}-{end} ({n_years} year{'s' if n_years != 1 else ''}). Data pulled {c['fetched'][:16]}."
    return label, kpis, f_out, f_cit, f_typ, table.to_dict("records")


# ------------------------------------------------- collaborations and expertise
SFU_NAMES = {"Simon Fraser University"}


@lru_cache(maxsize=None)
def cname(cc):
    if pycountry and cc:
        r = pycountry.countries.get(alpha_2=cc.upper())
        if r:
            return getattr(r, "common_name", None) or r.name
    return cc or "Unknown"


@lru_cache(maxsize=None)
def iso3(cc):
    if pycountry and cc:
        r = pycountry.countries.get(alpha_2=cc.upper())
        return r.alpha_3 if r else None
    return None


def selected_members(groups_sel, members_sel):
    c = corpus()
    base = {m for m in c["roster"] if c["groups"][m] in set(groups_sel or [])}
    if members_sel:
        base &= set(members_sel)
    return tuple(sorted(base))


def _members_label(ms, k=4):
    ms = sorted(ms)
    return "; ".join(ms[:k]) + (f" (+{len(ms) - k})" if len(ms) > k else "")


def person_rows(people):
    labels = corpus()["labels"]
    rows = [dict(collaborator=labels[k],
                 institution=p["insts"].most_common(1)[0][0] if p["insts"] else "Unknown",
                 country=cname(p["ctrs"].most_common(1)[0][0]) if p["ctrs"] else "Unknown",
                 works=len(p["works"]), siat_members=_members_label(p["members"]))
            for k, p in people.items()]
    return sorted(rows, key=lambda r: -r["works"])


def _new_person():
    return dict(works=set(), members=set(), insts=Counter(), ctrs=Counter())


@lru_cache(maxsize=1)
def _filter_options(stamp):
    c = corpus()
    cw, iw = defaultdict(set), defaultdict(set)
    for wid, d in c["ext"].items():
        for insts, ccs in d.values():
            for i in insts:
                iw[i].add(wid)
            for cc in ccs:
                cw[cc].add(wid)
    countries = sorted((cname(cc), cc) for cc in cw)
    top = [i for i in sorted(iw, key=lambda i: -len(iw[i])) if c["inst"][i][0] not in SFU_NAMES][:300]
    return ([{"label": n, "value": cc} for n, cc in countries],
            [{"label": f'{c["inst"][i][0]} ({len(iw[i])})', "value": i} for i in top])


@lru_cache(maxsize=32)
def collab_stats(stamp, start, end, max_authors, members, exclude_sfu):
    """External collaborations of the selected SIAT members. A joint work counts once per institution/country."""
    c = corpus()
    S, names = set(members), c["inst"]
    inst = defaultdict(lambda: dict(works=set(), people=set(), members=set()))
    ctry = defaultdict(lambda: dict(works=set(), people=set(), insts=set(), members=set()))
    people = defaultdict(_new_person)
    heat = defaultdict(lambda: defaultdict(set))
    joint, n_slots, n_aff = set(), 0, 0
    for wid in work_ids_in(c, start, end):
        ks = c["keys"][wid]
        if len(ks) > max_authors:
            continue
        mem = [k for k in ks if k in S]
        if not mem:
            continue
        for k, (insts, ccs) in c["ext"].get(wid, {}).items():
            n_slots += 1
            n_aff += bool(insts or ccs)
            joint.add(wid)
            keep = [i for i in insts if not (exclude_sfu and names[i][0] in SFU_NAMES)]
            sfu_only = exclude_sfu and bool(insts) and not keep
            for i in keep:
                d = inst[i]
                d["works"].add(wid); d["people"].add(k); d["members"].update(mem)
                for m in mem:
                    heat[m][i].add(wid)
            for cc in ccs:
                d = ctry[cc]
                d["works"].add(wid); d["people"].add(k); d["members"].update(mem)
                d["insts"].update(i for i in insts if names[i][1] == cc)
            if not sfu_only:
                p = people[k]
                p["works"].add(wid); p["members"].update(mem)
                p["insts"].update(names[i][0] for i in keep); p["ctrs"].update(ccs)

    inst_df = pd.DataFrame([dict(id=i, institution=names[i][0], country=cname(names[i][1]) if names[i][1] else "Unknown",
                                 type=names[i][2], works=len(d["works"]), collaborators=len(d["people"]),
                                 siat_members=len(d["members"])) for i, d in inst.items()],
                           columns=["id", "institution", "country", "type", "works", "collaborators", "siat_members"])
    inst_df = inst_df.sort_values(["works", "collaborators"], ascending=False)
    ctry_df = pd.DataFrame([dict(code=cc, country=cname(cc), works=len(d["works"]), collaborators=len(d["people"]),
                                 institutions=len(d["insts"]), siat_members=len(d["members"])) for cc, d in ctry.items()],
                           columns=["code", "country", "works", "collaborators", "institutions", "siat_members"])
    ctry_df = ctry_df.sort_values("works", ascending=False)
    top = inst_df.head(15).id.tolist()
    rows = sorted(heat, key=lambda m: -sum(len(heat[m].get(i, ())) for i in top))
    heat_df = pd.DataFrame([[len(heat[m].get(i, ())) for i in top] for m in rows], index=rows,
                           columns=[names[i][0] for i in top])
    return dict(inst=inst_df, country=ctry_df, people=person_rows(people), heat=heat_df,
                n_slots=n_slots, n_aff=n_aff, n_works=len(joint))


def empty_fig(msg):
    fig = go.Figure()
    fig.update_layout(height=300, xaxis_visible=False, yaxis_visible=False, margin=dict(t=20, b=20),
                      annotations=[dict(text=msg, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")])
    return fig


def heatmap_fig(df, title):
    if df.empty or df.shape[1] == 0:
        return empty_fig("No data for this selection")
    z = df.values.astype(float)
    z[z == 0] = np.nan
    text = [["" if np.isnan(v) else str(int(v)) for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, x=list(df.columns), y=list(df.index), text=text, texttemplate="%{text}",
                               colorscale="Reds", hoverongaps=False, showscale=False,
                               hovertemplate="%{y}<br>%{x}<br>works: %{z}<extra></extra>"))
    fig.update_layout(title=title, height=max(300, 26 * len(df) + 190), margin=dict(t=40, b=10, l=10, r=10),
                      xaxis=dict(tickangle=-35, automargin=True), yaxis=dict(autorange="reversed", automargin=True))
    return fig


def country_map(df, exclude_canada):
    d = df[df.code != "CA"] if exclude_canada else df
    d = d.assign(iso=d.code.map(iso3)).dropna(subset=["iso"])
    if d.empty:
        return empty_fig("No country data (is pycountry installed?)" if not pycountry else "No country data")
    z = np.log10(d.works.astype(float))
    top = int(math.ceil(z.max())) if z.max() > 0 else 1
    fig = go.Figure(go.Choropleth(
        locations=d.iso, z=z, colorscale="Reds", marker_line_width=0.3,
        customdata=d[["country", "works", "collaborators", "institutions"]].values,
        hovertemplate="<b>%{customdata[0]}</b><br>joint works: %{customdata[1]}<br>"
                      "collaborators: %{customdata[2]}<br>institutions: %{customdata[3]}<extra></extra>",
        colorbar=dict(title="Joint works", tickvals=list(range(top + 1)), ticktext=[str(10 ** i) for i in range(top + 1)])))
    fig.update_geos(showframe=False, projection_type="natural earth", showcountries=True, countrycolor="#ccc")
    fig.update_layout(title="Joint works by collaborator country (log scale)" + (", excluding Canada" if exclude_canada else ""),
                      height=480, margin=dict(t=40, b=0, l=0, r=0))
    return fig


def collab_view(n_years, end_year, groups_sel, members_sel, max_auth, opts, top_n):
    start, end = window(n_years, end_year)
    S = selected_members(groups_sel, members_sel)
    if not S:
        e = empty_fig("Select at least one faculty group")
        return "No members selected.", e, [], e, [], e, []
    st = collab_stats(_stamp(), start, end, max_auth, S, "nosfu" in opts)
    inst, ctry = st["inst"], st["country"]
    pct = 100 * st["n_aff"] / st["n_slots"] if st["n_slots"] else 0
    summary = (f"{len(S)} SIAT members, {start}-{end}: {st['n_works']} works with external co-authors; "
               f"{len(inst)} institutions in {len(ctry)} countries; {len(st['people'])} external collaborators. "
               f"OpenAlex has no affiliation for {st['n_slots'] - st['n_aff']} of {st['n_slots']} external author "
               f"slots ({100 - pct:.0f}%); those authors are missing from institution and country views.")
    top = inst.head(top_n).iloc[::-1]
    if top.empty:
        bar = empty_fig("No institutions for this selection")
    else:
        bar = px.bar(top, x="works", y="institution", orientation="h", hover_data=["country", "collaborators", "siat_members"],
                     title=f"Top {len(top)} institutions by joint works")
        bar.update_traces(marker_color=RED)
        bar.update_layout(height=max(300, 24 * len(top) + 90), margin=dict(t=40, b=30, l=10, r=10),
                          xaxis_title="Joint works", yaxis_title=None, yaxis=dict(automargin=True))
    return (summary, bar, inst.drop(columns="id").to_dict("records"), country_map(ctry, "nocan" in opts),
            ctry.drop(columns="code").to_dict("records"),
            heatmap_fig(st["heat"], "Joint works: SIAT member by institution (top 15)"), st["people"][:500])


@lru_cache(maxsize=32)
def topic_rows(stamp, start, end, max_authors, members, scope, country, inst_id):
    """One row per (work, selected SIAT member), using OpenAlex's primary topic for each work."""
    c = corpus()
    S = set(members)
    need_ext = scope == "collab" or bool(country) or bool(inst_id)
    out = []
    for wid in work_ids_in(c, start, end):
        t = c["topic"].get(wid)
        if not t:
            continue
        ks = c["keys"][wid]
        mem = [k for k in ks if k in S]
        if not mem:
            continue
        if need_ext:
            ex = c["ext"].get(wid, {})
            if len(ks) > max_authors or not ex:
                continue
            if country and not any(country in ccs for _, ccs in ex.values()):
                continue
            if inst_id and not any(inst_id in insts for insts, _ in ex.values()):
                continue
        out.extend((wid, m) + t for m in mem)
    return pd.DataFrame(out, columns=["wid", "member", "domain", "field", "subfield", "topic"])


def topic_view(n_years, end_year, groups_sel, members_sel, max_auth, scope, country, inst_id):
    c = corpus()
    start, end = window(n_years, end_year)
    S = selected_members(groups_sel, members_sel)
    e = empty_fig("No data for this selection")
    if not c["topic"]:
        return "No topic data in works.json. Re-run the collection notebook (it now fetches topics).", e, e, []
    rows = topic_rows(_stamp(), start, end, max_auth, S, scope, country, inst_id)
    if rows.empty:
        return "No works with a topic classification match this selection.", e, e, []
    works = rows.drop_duplicates("wid")
    filtered = scope == "collab" or country or inst_id
    cover = ""
    if not filtered:
        d = c["df"][c["df"].year.between(start, end)]
        total = sum(1 for ms in d.members if S and set(S) & set(ms))
        cover = f" of {total} works in the window ({100 * len(works) / max(total, 1):.0f}%)"
    summary = (f"{len(works)} works{cover} have an OpenAlex topic and are shown (each work counted once, by its primary topic). "
               f"{works.subfield.nunique()} subfields across {works.field.nunique()} fields.")
    tdf = works.groupby(["field", "subfield", "topic"]).size().reset_index(name="works")
    tree = px.treemap(tdf, path=[px.Constant("All works"), "field", "subfield", "topic"], values="works",
                      color="field", maxdepth=3, title="Topic map (click a block to zoom in, click the header to zoom out)")
    tree.update_layout(height=620, margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
    top_sub = rows.drop_duplicates("wid").groupby("subfield").size().nlargest(15).index
    piv = (rows[rows.subfield.isin(top_sub)].groupby(["member", "subfield"]).wid.nunique()
           .unstack(fill_value=0).reindex(columns=list(top_sub)))
    piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]
    heat = heatmap_fig(piv, "Who works on what: SIAT member by subfield (top 15 subfields)")
    table = (works.groupby(["topic", "subfield", "field"]).wid.nunique().reset_index(name="works")
             .merge(rows.groupby(["topic"]).member.nunique().reset_index(name="siat_members"), on="topic")
             .sort_values("works", ascending=False))
    return summary, tree, heat, table.to_dict("records")


def finder_options(n_years, end_year, groups_sel, members_sel, max_auth):
    c = corpus()
    if not c["topic"]:
        return []
    start, end = window(n_years, end_year)
    rows = topic_rows(_stamp(), start, end, max_auth, selected_members(groups_sel, members_sel), "all", None, None)
    cnt = rows.drop_duplicates("wid").groupby("subfield").size().sort_values(ascending=False)
    return [{"label": f"{s} ({k})", "value": s} for s, k in cnt.items()]


def finder_view(n_years, end_year, groups_sel, members_sel, max_auth, subfield):
    if not subfield:
        return [], []
    c = corpus()
    start, end = window(n_years, end_year)
    S = selected_members(groups_sel, members_sel)
    rows = topic_rows(_stamp(), start, end, max_auth, S, "all", None, None)
    r = rows[rows.subfield == subfield]
    mem = (r.groupby("member").wid.nunique().reset_index(name="works").sort_values("works", ascending=False))
    mem["group"] = mem.member.map(c["groups"])
    people, names = defaultdict(_new_person), c["inst"]
    for wid in set(r.wid):
        ks = c["keys"][wid]
        if len(ks) > max_auth:
            continue
        present = [k for k in ks if k in S]
        for k, (insts, ccs) in c["ext"].get(wid, {}).items():
            p = people[k]
            p["works"].add(wid); p["members"].update(present)
            p["insts"].update(names[i][0] for i in insts); p["ctrs"].update(ccs)
    return mem[["member", "group", "works"]].to_dict("records"), person_rows(people)[:200]


# ------------------------------------------------------------------------- layout
CARD = {"border": "1px solid #e3e6ea", "borderRadius": "8px", "padding": "12px 16px", "background": "#fff"}
GRID = {"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(230px,1fr))", "gap": "14px",
        "alignItems": "start", "margin": "10px 0"}


def ctl(label, comp):
    return html.Div([html.Div(label, style={"fontSize": "12px", "color": "#555", "marginBottom": "4px"}), comp])


def tip(**kw):
    return dict(tooltip={"always_visible": True, "placement": "bottom"}, **kw)


def legend():
    dot = lambda col: html.Span(style={"display": "inline-block", "width": "11px", "height": "11px",
                                       "borderRadius": "50%", "background": col, "marginRight": "5px"})
    items = list(GROUP_COLORS.items()) + [("Non-SIAT author", GREY)]
    return html.Div([html.Span([dot(col), name], style={"marginRight": "18px"}) for name, col in items],
                    style={"fontSize": "13px", "margin": "4px 0"})


def graph_section(pfx, title, blurb, roster_names, directed):
    opts = [{"label": " Include non-SIAT authors", "value": "ext"},
            {"label": " Show links between non-SIAT authors", "value": "extext"},
            {"label": " Label non-SIAT authors", "value": "labels"}]
    return html.Div([
        html.H2(title, style={"marginBottom": 2}),
        html.Div(blurb, style={"color": "#666", "fontSize": "13px"}),
        html.Div([
            ctl("Display", dcc.Checklist(id=f"{pfx}-opts", options=opts, value=["ext"])),
            ctl("Min. ties to SIAT (non-SIAT authors)", dcc.Slider(1, 10, 1, value=2, id=f"{pfx}-minties", **tip())),
            ctl("Max. non-SIAT authors shown", dcc.Slider(5, 200, 5, value=40, id=f"{pfx}-topn", **tip())),
            ctl("Min. edge weight", dcc.Slider(1, 10, 1, value=1, id=f"{pfx}-minedge", **tip())),
            ctl("Skip works with more authors than", dcc.Slider(5, 100, 5, value=25, id=f"{pfx}-maxauth", **tip())),
            ctl("Focus on one member (ego network)",
                dcc.Dropdown(id=f"{pfx}-focus", options=roster_names, placeholder="All members", clearable=True)),
            ctl("Layout", dcc.Dropdown(id=f"{pfx}-layout", clearable=False, value="cose", options=[
                {"label": "Force-directed", "value": "cose"}, {"label": "Concentric (by degree)", "value": "concentric"},
                {"label": "Circle", "value": "circle"}, {"label": "Grid", "value": "grid"}])),
            ctl("Node size", dcc.Dropdown(id=f"{pfx}-size", clearable=False, value="scaled", options=[
                {"label": "Scaled (works, or citations received)", "value": "scaled"},
                {"label": "Equal for SIAT members", "value": "equal"}])),
            ctl("Export", html.Button("Download PNG", id=f"{pfx}-png")),
        ], style=GRID),
        legend(),
        html.Div(id=f"{pfx}-stats", style={"fontSize": "13px", "margin": "4px 0"}),
        cyto.Cytoscape(id=f"{pfx}-cy", elements=[], layout=LAYOUTS["cose"], minZoom=0.1, maxZoom=3,
                       style={"width": "100%", "height": "680px", "border": "1px solid #e3e6ea", "background": "#fff"}),
        html.Div(id=f"{pfx}-info", children="Click a node for details.",
                 style={**CARD, "marginTop": "8px", "fontSize": "13px"}),
    ], style={"marginTop": "36px"})


def dtable(id_, cols, page_size=12):
    return dash_table.DataTable(
        id=id_, sort_action="native", export_format="csv", page_size=page_size,
        columns=[{"name": n, "id": i} for n, i in cols], style_header={"fontWeight": 600},
        style_cell={"padding": "6px", "textAlign": "left", "maxWidth": 280, "overflow": "hidden",
                    "textOverflow": "ellipsis"},
        style_table={"overflowX": "auto"})


def expertise_section(roster_names, country_opts, inst_opts):
    muted = {"fontSize": "13px", "color": "#555", "margin": "4px 0"}
    two = {"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(420px,1fr))", "gap": "12px"}
    return html.Div([
        html.H2("Collaborations and expertise", style={"marginBottom": 2}),
        html.Div("Who SIAT faculty collaborate with, where, and on what. The filters in this block apply to everything in it "
                 "(not to the network graphs below). Collaborators are authors who are not on the SIAT roster; a joint work is a "
                 "work with at least one selected SIAT member and at least one author from that institution or country, counted "
                 "once per work. Connection strength is the number of joint works.", style=muted),
        html.Div([
            ctl("Faculty groups", dcc.Checklist(id="f-groups", value=list(GROUP_COLORS),
                                                options=[{"label": " " + g, "value": g} for g in GROUP_COLORS])),
            ctl("Members (empty = all)", dcc.Dropdown(id="f-members", options=roster_names, multi=True,
                                                      placeholder="All members")),
            ctl("Skip works with more authors than", dcc.Slider(5, 200, 5, value=50, id="f-maxauth", **tip())),
        ], style=GRID),

        html.H3("Collaborating institutions and countries", style={"marginTop": 20}),
        html.Div([
            ctl("Display", dcc.Checklist(id="c-opts", value=["nosfu", "nocan"], options=[
                {"label": " Exclude SFU from institution and collaborator lists", "value": "nosfu"},
                {"label": " Exclude Canada from the map", "value": "nocan"}])),
            ctl("Institutions in bar chart", dcc.Slider(5, 50, 5, value=20, id="c-topn", **tip())),
        ], style=GRID),
        html.Div(id="c-summary", style=muted),
        html.Div([dcc.Graph(id="c-inst-bar"), dcc.Graph(id="c-map")], style=two),
        html.Div([html.Div([html.H4("Institutions"),
                            dtable("c-inst-table", [("Institution", "institution"), ("Country", "country"), ("Type", "type"),
                                                    ("Joint works", "works"), ("Collaborators", "collaborators"),
                                                    ("SIAT members", "siat_members")])]),
                  html.Div([html.H4("Countries"),
                            dtable("c-country-table", [("Country", "country"), ("Joint works", "works"),
                                                       ("Collaborators", "collaborators"), ("Institutions", "institutions"),
                                                       ("SIAT members", "siat_members")])])], style=two),
        dcc.Graph(id="c-heat"),
        html.H4("External collaborators"),
        dtable("c-people-table", [("Collaborator", "collaborator"), ("Institution", "institution"), ("Country", "country"),
                                  ("Joint works", "works"), ("SIAT members", "siat_members")], page_size=15),

        html.H3("Topics and expertise", style={"marginTop": 28}),
        html.Div("Topics are OpenAlex's primary topic for each work, grouped as field, subfield and topic. Choose a country or "
                 "institution to see the topics of the work SIAT shares with them.", style=muted),
        html.Div([
            ctl("Scope", dcc.RadioItems(id="t-scope", value="all", options=[
                {"label": " All works", "value": "all"}, {"label": " Only works with external co-authors", "value": "collab"}])),
            ctl("Collaborator country", dcc.Dropdown(id="t-country", options=country_opts, placeholder="Any country")),
            ctl("Collaborator institution (top 300)", dcc.Dropdown(id="t-inst", options=inst_opts, placeholder="Any institution")),
        ], style=GRID),
        html.Div(id="t-summary", style=muted),
        dcc.Graph(id="t-tree"),
        dcc.Graph(id="t-heat"),
        html.H4("Topics"),
        dtable("t-table", [("Topic", "topic"), ("Subfield", "subfield"), ("Field", "field"), ("Works", "works"),
                           ("SIAT members", "siat_members")], page_size=15),

        html.H3("Topic finder", style={"marginTop": 28}),
        html.Div("Pick a subfield to see which SIAT members work in it and who they have co-authored with, as a starting "
                 "point for introductions.", style=muted),
        ctl("Subfield", dcc.Dropdown(id="t-finder", placeholder="Choose a subfield")),
        html.Div([html.Div([html.H4("SIAT members"),
                            dtable("fd-mem", [("Member", "member"), ("Group", "group"), ("Works", "works")])]),
                  html.Div([html.H4("External co-authors on these works"),
                            dtable("fd-ext", [("Collaborator", "collaborator"), ("Institution", "institution"),
                                              ("Country", "country"), ("Joint works", "works"),
                                              ("SIAT members", "siat_members")])])], style=two),
    ], style={"marginTop": "36px"})


def serve_layout():
    try:
        c = corpus()
    except FileNotFoundError:
        return html.Div(f"Could not find {WORKS_FILE} and {ROSTER_FILE}. Run the collection notebook first, "
                        "or set SIAT_DATA to the folder that holds them.", style={"padding": 24})
    this_year = date.today().year
    min_y, max_y = int(c["df"].year.min()), max(int(c["df"].year.max()), this_year)
    end_default = this_year if this_year <= max_y else max_y
    span = max(2, end_default - min_y + 1)
    marks = {i: str(i) for i in range(5, span, 5)}
    marks.update({1: "1", span: "All"})
    roster_names = [{"label": n, "value": n} for n in sorted(c["roster"])]
    country_opts, inst_opts = _filter_options(_stamp())
    return html.Div([
        html.Div([
            html.H1("SIAT research dashboard", style={"margin": "0 0 6px"}),
            html.Div([
                html.Div(ctl("Years to include (counting back from end year)",
                             dcc.Slider(1, span, 1, value=min(DEFAULT_YEARS, span), marks=marks, id="years-n", **tip())),
                         style={"flex": "3", "minWidth": "300px"}),
                html.Div(ctl("End year", dcc.Dropdown(id="year-end", clearable=False, value=end_default,
                                                      options=list(range(max_y, min_y - 1, -1)))),
                         style={"flex": "1", "minWidth": "120px"}),
            ], style={"display": "flex", "gap": "24px", "alignItems": "start"}),
            html.Div(id="window-label", style={"fontSize": "13px", "color": "#555"}),
        ], style={"position": "sticky", "top": 0, "zIndex": 100, "background": "#f7f8fa",
                  "padding": "10px 0 6px", "borderBottom": "1px solid #e3e6ea"}),
        html.Div(id="kpis", style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(180px,1fr))",
                                   "gap": "14px", "margin": "16px 0"}),
        html.Div([dcc.Graph(id="fig-out"), dcc.Graph(id="fig-cit"), dcc.Graph(id="fig-typ")],
                 style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit,minmax(340px,1fr))", "gap": "10px"}),
        html.Div("Citations are OpenAlex's all-time counts for works published in the window.",
                 style={"fontSize": "12px", "color": "#666"}),
        dash_table.DataTable(id="authors-table", sort_action="native", export_format="csv", page_size=30,
                             columns=[{"name": n, "id": i} for n, i in
                                      [("Author", "author"), ("Group", "group"), ("Outputs", "outputs"),
                                       ("Citations", "citations"), ("h-index (window)", "h_index")]],
                             style_header={"fontWeight": 600}, style_cell={"padding": "6px", "textAlign": "left"}),
        expertise_section(roster_names, country_opts, inst_opts),
        graph_section("co", "Co-authorship network",
                      "Nodes are authors (size = works in window); edges are joint works.", roster_names, False),
        graph_section("ci", "Citation network",
                      "Arrows run from citing author to cited author, among works in the collected corpus "
                      "(works with at least one SIAT author). Self-citations are dropped. Node size = citations received "
                      "within the corpus.", roster_names, True),
    ], style={"maxWidth": "1300px", "margin": "0 auto", "padding": "0 18px 60px",
              "fontFamily": "system-ui, -apple-system, Segoe UI, sans-serif"})


app = Dash(__name__, title="SIAT dashboard")
server = app.server
app.layout = serve_layout  # a function, so a page refresh re-reads the data


@app.callback(Output("window-label", "children"), Output("kpis", "children"), Output("fig-out", "figure"),
              Output("fig-cit", "figure"), Output("fig-typ", "figure"), Output("authors-table", "data"),
              Input("years-n", "value"), Input("year-end", "value"))
def _overview(n, end):
    return overview_view(n, end)


def _register(pfx, directed):
    inputs = [Input("years-n", "value"), Input("year-end", "value"), Input(f"{pfx}-opts", "value"),
              Input(f"{pfx}-minties", "value"), Input(f"{pfx}-topn", "value"), Input(f"{pfx}-minedge", "value"),
              Input(f"{pfx}-maxauth", "value"), Input(f"{pfx}-focus", "value"), Input(f"{pfx}-layout", "value"),
              Input(f"{pfx}-size", "value")]

    @app.callback(Output(f"{pfx}-cy", "elements"), Output(f"{pfx}-cy", "stylesheet"),
                  Output(f"{pfx}-cy", "layout"), Output(f"{pfx}-stats", "children"), *inputs)
    def _graph(*args):
        return graph_view(directed, *args)

    @app.callback(Output(f"{pfx}-info", "children"), Input(f"{pfx}-cy", "tapNodeData"))
    def _info(d):
        if not d:
            return "Click a node for details."
        return html.Div([html.B(d["label"]), f" ({d['kind']}). {d['value_label']}: {d['value']}. ",
                         html.Div(f"Strongest links: {d['nbrs'] or 'none shown'}", style={"marginTop": "4px"})])

    @app.callback(Output(f"{pfx}-cy", "generateImage"), Input(f"{pfx}-png", "n_clicks"), prevent_initial_call=True)
    def _png(_):
        return {"type": "png", "action": "download", "filename": f"siat_{pfx}_network"}


COMMON = [Input("years-n", "value"), Input("year-end", "value"), Input("f-groups", "value"),
          Input("f-members", "value"), Input("f-maxauth", "value")]


@app.callback(Output("c-summary", "children"), Output("c-inst-bar", "figure"), Output("c-inst-table", "data"),
              Output("c-map", "figure"), Output("c-country-table", "data"), Output("c-heat", "figure"),
              Output("c-people-table", "data"), *COMMON, Input("c-opts", "value"), Input("c-topn", "value"))
def _collab(n, end, groups, members, max_auth, opts, top_n):
    return collab_view(n, end, groups, members, max_auth, opts or [], top_n)


@app.callback(Output("t-summary", "children"), Output("t-tree", "figure"), Output("t-heat", "figure"),
              Output("t-table", "data"), *COMMON, Input("t-scope", "value"), Input("t-country", "value"),
              Input("t-inst", "value"))
def _topics(n, end, groups, members, max_auth, scope, country, inst_id):
    return topic_view(n, end, groups, members, max_auth, scope, country, inst_id)


@app.callback(Output("t-finder", "options"), *COMMON)
def _finder_options(n, end, groups, members, max_auth):
    return finder_options(n, end, groups, members, max_auth)


@app.callback(Output("fd-mem", "data"), Output("fd-ext", "data"), *COMMON, Input("t-finder", "value"))
def _finder(n, end, groups, members, max_auth, subfield):
    return finder_view(n, end, groups, members, max_auth, subfield)


_register("co", False)
_register("ci", True)

if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", 8050)))
