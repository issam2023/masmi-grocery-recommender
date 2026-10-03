"""Run with: python -m streamlit run app.py"""
import json
from pathlib import Path
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="Masmi | Grocery recommender v1", page_icon="🛒", layout="wide")
st.markdown('''<style>
.stApp {background: #f7f8f4; color: #173e34;}
.stApp p, .stApp label {color: #173e34;}
.hero p {color: #dbe9df !important;}
.block-container {max-width: 1180px; padding-top: 2.4rem;}
h1,h2,h3 {color:#173e34;}
div[data-testid="stMetric"] {background:white; border:1px solid #e1e8df; border-radius:16px; padding:18px;}
.hero {background:#173e34; color:#fff; padding:32px; border-radius:22px; margin-bottom:24px;}
.hero h1 {color:white; margin:0; font-size:2.6rem;}
.hero p {color:#dbe9df; font-size:1.08rem;}
.tag {font-size:12px; letter-spacing:2px; color:#c8d8aa;}
footer {visibility:hidden;}
</style>''', unsafe_allow_html=True)
ROOT = Path(__file__).resolve().parent

@st.cache_data
def load_data(rule_path, csv_path, rule_stamp, csv_stamp):
    payload = json.loads(Path(rule_path).read_text(encoding="utf-8"))
    rules = payload["rules"]
    for r in rules:
        r["antecedents"] = frozenset(r["antecedents"])
        r["consequents"] = frozenset(r["consequents"])
    df = pd.read_csv(csv_path)
    df["Item"] = df["Item"].str.strip()
    pairs = df[["TransactionID", "Item"]].drop_duplicates()
    counts = pairs.groupby("Item").size().sort_values(ascending=False)
    frequency = counts / pairs.TransactionID.nunique()
    return rules, payload.get("metadata", {}), sorted(frequency.index), frequency, pairs.TransactionID.nunique()

def recommend(basket, rules, top_n=3, min_lift=1.0, single_only=False):
    basket = set(basket)
    best = {}
    for r in rules:
        if r["lift"] <= min_lift or (single_only and len(r["consequents"]) != 1):
            continue
        if not r["antecedents"].issubset(basket):
            continue
        for item in r["consequents"] - basket:
            candidate = {"item":item, "lift":float(r["lift"]), "confidence":float(r["confidence"]),
                         "support":float(r["support"]), "because":sorted(r["antecedents"]),
                         "bundle":sorted(r["consequents"])}
            if item not in best or (candidate["lift"],candidate["confidence"]) > (best[item]["lift"],best[item]["confidence"]):
                best[item] = candidate
    return sorted(best.values(),key=lambda r:(-r["lift"],-r["confidence"],r["item"]))[:top_n]

rule_path, csv_path = ROOT / "association_rules.json", ROOT / "transactions.csv"
missing = [p.name for p in [rule_path, csv_path] if not p.is_file()]
if missing:
    st.error("Place these files beside app.py: " + ", ".join(missing))
    st.stop()
try:
    rules, metadata, catalogue, popularity, transaction_count = load_data(str(rule_path), str(csv_path), rule_path.stat().st_mtime_ns, csv_path.stat().st_mtime_ns)
except (ValueError, KeyError, TypeError, OSError) as exc:
    st.error(f"Could not read the saved data: {exc}")
    st.stop()

st.markdown('<style>\n.masmi-banner {\ndisplay:flex; align-items:center; gap:32px; flex-wrap:wrap;\nbackground:linear-gradient(120deg,#102b50,#164b7a);\npadding:32px; border-radius:24px; margin-bottom:24px;\n}\n.masmi-circle {\nwidth:180px; height:180px; flex-shrink:0;\nborder-radius:50%; background:#1769e0;\nborder:5px solid #8cc8ff;\nbox-shadow:0 8px 24px #00000040;\ndisplay:flex; flex-direction:column;\nalign-items:center; justify-content:center;\ntext-align:center;\n}\n.masmi-circle span {\ncolor:#ffffff !important;\nfont-size:24px; font-weight:800; line-height:1.3;\n}\n.masmi-circle small {\ncolor:#ffffff !important;\nfont-size:13px; margin-bottom:10px; letter-spacing:2px;\n}\n.masmi-banner h1 {\ncolor:#ffffff !important;\nfont-size:clamp(26px,3vw,40px); line-height:1.15;\n}\n.masmi-banner p {\ncolor:#e0efff !important; font-size:17px;\n}\n</style>\n<div class="masmi-banner">\n<div class="masmi-circle">\n<small>CREATED BY</small>\n<span>Abdelhafid<br>Masmi</span>\n</div>\n<div style="flex:1; min-width:220px;">\n<h1>Find your next grocery pick</h1>\n<p>Build your basket. Find products that go together.</p>\n</div>\n</div>', unsafe_allow_html=True)
st.caption("v1 · Synthetic grocery data · Association rules · Abdelhafid Masmi")
a,b,c = st.columns(3)
a.metric("Shopping baskets", f"{transaction_count:,}")
b.metric("Products", len(catalogue))
c.metric("Saved rules", f"{len(rules):,}")

with st.sidebar:
    st.title("Your grocery assistant")
    top_n = st.slider("How many suggestions?", 1, 8, 3)
    with st.expander("Advanced recommendation settings"):
        mode = st.radio("Recommendation mode", ["Single-product suggestions", "Homework rules"], help="Single-product mode scores each suggested product individually.")
        min_lift = st.slider("Minimum lift (exclusive)", 1.0, 5.0, 1.0, 0.1)
        min_confidence = st.slider("Minimum confidence", 0.30, 1.0, 0.30, 0.05)
        min_support = st.slider("Minimum support", 0.02, 0.20, 0.02, 0.01)
        st.caption("These controls filter saved rules. They do not retrain the system.")
    st.divider()
    st.write("1. Choose your products\n\n2. See what goes well together\n\n3. Add a suggestion")
    st.caption("Practice app using synthetic receipts. Suggestions show shopping patterns, not guaranteed preferences.")

tab_basket, tab_learn = st.tabs(["🛒 Build my basket", "📖 Understand the recommendations"])
with tab_basket:
    st.write("Choose what you already have, then add any suggestions you like. Your basket updates immediately.")
    left_basket, right_suggestions = st.columns([1,1], gap="large")
    with left_basket:
        st.subheader("Your basket")
        if "basket" not in st.session_state:
            st.session_state.basket = []

        def set_basket(items):
            st.session_state.pop("last_applied", None)
            st.session_state.basket = [i for i in items if i in catalogue]

        examples = st.columns(2)
        for col,label,items in zip([examples[0]]*5,["🔥 Barbecue","👶 Baby essentials","🍝 Pasta night","☕ Breakfast","Clear basket"],[["Charcoal"],["Diapers"],["Pasta","Garlic"],["Coffee"],[]]):
            examples[["🔥 Barbecue","👶 Baby essentials","🍝 Pasta night","☕ Breakfast","Clear basket"].index(label) % 2].button(label,on_click=set_basket,args=(items,),width="stretch")
        basket = st.multiselect("Products in your basket", catalogue, key="basket", placeholder="Choose one or more products…")
        st.caption(f"{len(basket)} product(s) selected. Suggestions exclude products already in your basket.")

        st.markdown("**Or browse products**")
        product_search = st.text_input("Find a product", placeholder="Search coffee, bread, milk…", key="browse_search")
        available = [i for i in catalogue if i not in basket and product_search.casefold() in i.casefold()]
        if available:
            st.caption("Showing up to 8 products. Use the basket selector above to find any catalogue item.")
            for item in available[:8]:
                row = st.columns([3,1])
                row[0].write(item)
                row[1].button("Add",key="browse_"+item,on_click=set_basket,args=(basket+[item],),width="stretch")
        else:
            st.caption("No unselected products match your search.")
    # Restrict the loaded rules without mining again.
    active_rules = [r for r in rules if r["confidence"] >= min_confidence and r["support"] >= min_support]
    matching = [r for r in active_rules if r["lift"] > min_lift
                and r["antecedents"].issubset(set(basket))
                and bool(r["consequents"] - set(basket))
                and (mode != "Single-product suggestions" or len(r["consequents"]) == 1)]
    matching.sort(key=lambda r:(-r["lift"], -r["confidence"], sorted(r["antecedents"]), sorted(r["consequents"])))


    with right_suggestions:
        results = recommend(basket,active_rules,top_n,min_lift,mode == "Single-product suggestions")
        st.subheader("Suggestions for you")
        if results:
            st.write("Add one suggestion using its button, or add the full recommended list.")
            st.button("Add all recommended products", type="primary", on_click=set_basket,
                      args=(basket+[r["item"] for r in results],), width="stretch")
            for start in range(0,len(results),1):
                cols = st.columns(1)
                for col,r in zip(cols,results[start:start+1]):
                    with col:
                        with st.container(border=True):
                            st.subheader(r["item"])
                            st.caption("Pairs with " + ", ".join(r["because"]))
                            st.write("Customers buying **" + ", ".join(r["because"]) + "** also bought this product.")
                        with st.expander("Why this suggestion?", expanded=False):
                            st.write(f"In the matching historical baskets, {r['confidence']:.1%} also contain the suggested product or bundle. The rule's lift is {r['lift']:.2f}× and support is {r['support']:.1%}.")
                            st.caption("Based on saved co-purchase patterns.")
                            if len(r["bundle"])>1:
                                st.caption("Score belongs to the bundle: " + ", ".join(r["bundle"]))
                            st.button("Add to basket", key="add_"+r["item"],on_click=set_basket,args=(basket+[r["item"]],),width="stretch")
        else:
            if basket:
                st.info("No association rule matches this basket and the selected filters. Here are some popular products instead.")
            else:
                st.info("Choose a product to get matching recommendations. Start with these popular picks.")
            fallback = [i for i in popularity.index if i not in basket and i != "Reusable Bag"][:top_n]
            st.caption("Popular products · These are not personalized association recommendations.")
            for start in range(0,len(fallback),1):
                cols = st.columns(1)
                for col,item in zip(cols,fallback[start:start+1]):
                    with col:
                        with st.container(border=True):
                            st.subheader(item)
                            st.write(f"In **{popularity[item]:.1%}** of baskets")
                            st.button("Add to basket",key="popular_"+item,on_click=set_basket,args=(basket+[item],),width="stretch")

        with st.expander("Understand the recommendation scores"):
            st.write("Support is the share of all baskets containing the whole rule. Confidence is the share of baskets with the starting products that also contain the suggested products. Lift compares that confidence with the suggested products' baseline frequency.")
            st.write("Homework mode follows the notebook: a rule with multiple suggested products gives its whole-bundle score to each item. This can make a common item such as Reusable Bag inherit a strong bundle lift. Select Single-product suggestions to avoid that effect.")
            st.write("These patterns come from synthetic data and describe association, not cause and effect.")
        with st.expander("Explore the saved rules"):
            search = st.text_input("Find a product in a rule", placeholder="For example: Diapers")
            preview = []
            for r in sorted(rules,key=lambda x:-x["lift"]):
                left, right = ", ".join(sorted(r["antecedents"])), ", ".join(sorted(r["consequents"]))
                if search and search.casefold() not in (left+" "+right).casefold():
                    continue
                preview.append({"Starting products":left,"Suggested products":right,"Support":r["support"],"Confidence":r["confidence"],"Lift":r["lift"]})
                if len(preview)>=100:
                    break
            st.dataframe(pd.DataFrame(preview),hide_index=True,width="stretch")
            st.caption("Showing up to 100 rules, ranked by lift.")
            st.download_button("Download saved rules JSON",data=rule_path.read_bytes(),file_name="association_rules.json",mime="application/json")

        with st.expander("Advanced: inspect matching rules"):
            st.subheader("Inspect and apply a rule")
            if matching:
                labels = []
                for idx, rule in enumerate(matching):
                    left = ", ".join(sorted(rule["antecedents"]))
                    right = ", ".join(sorted(rule["consequents"]))
                    labels.append(f"{left} → {right} | lift {rule['lift']:.2f} | confidence {rule['confidence']:.0%}")
                selected_index = st.selectbox("Rules that can be applied to this basket", range(len(matching)), format_func=lambda idx:labels[idx])
                selected_rule = matching[selected_index]
                new_items = sorted(selected_rule["consequents"] - set(basket))
                with st.container(border=True):
                    st.markdown("**IF the basket contains:** " + ", ".join(sorted(selected_rule["antecedents"])))
                    st.markdown("**THEN suggest:** " + ", ".join(new_items))
                    x,y,z = st.columns(3)
                    x.metric("Lift", f"{selected_rule['lift']:.2f}×")
                    y.metric("Confidence", f"{selected_rule['confidence']:.1%}")
                    z.metric("Supporting baskets", f"{round(selected_rule['support']*transaction_count):,}")
                    st.caption(f"This rule occurs in {selected_rule['support']:.1%} of all baskets. Confidence describes observed co-purchases, not a guaranteed customer response.")
                    def apply_selected(current, suggested, label):
                        set_basket(current + suggested)
                        st.session_state.last_applied = label
                    st.button("Apply this rule · add " + ", ".join(new_items), type="primary", on_click=apply_selected,
                              args=(basket,new_items,labels[selected_index]), width="stretch")
                with st.expander(f"View all {len(matching):,} matching rules"):
                    rows = [{"If basket contains":", ".join(sorted(r["antecedents"])),
                             "Then suggest":", ".join(sorted(r["consequents"])),
                             "Lift":round(r["lift"],3),"Confidence (%)":round(r["confidence"]*100,2),
                             "Support (%)":round(r["support"]*100,2)} for r in matching]
                    st.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch")
            else:
                st.info("Choose an example basket above, or lower your filters, to find a matching rule.")
            if st.session_state.get("last_applied"):
                st.success("Last applied: " + st.session_state.last_applied)
                st.caption("The added product is now in your basket. Matching rules and recommendations have been recalculated.")

def bar_chart(frame, category, value, title, colour="#1769e0"):
    chart = alt.Chart(frame).mark_bar(color=colour, cornerRadiusEnd=3).encode(
        x=alt.X(value+":Q", title=title),
        y=alt.Y(category+":N", sort="-x", title=None),
        tooltip=[alt.Tooltip(category+":N"), alt.Tooltip(value+":Q", format=".2f")]
    ).properties(height=max(220,len(frame)*28))
    st.altair_chart(chart, width="stretch")

# Analytics use the saved dataset and rules. No Apriori run is triggered here.
rule_frame = pd.DataFrame([{
    "Starting products": ", ".join(sorted(r["antecedents"])),
    "Suggested products": ", ".join(sorted(r["consequents"])),
    "Support (%)": r["support"]*100,
    "Confidence (%)": r["confidence"]*100,
    "Lift": r["lift"],
    "Antecedent size": len(r["antecedents"]),
    "Consequent size": len(r["consequents"])
} for r in rules])
frequency_frame = (popularity*100).rename("Baskets (%)").rename_axis("Product").reset_index()

with tab_learn, st.expander("Shopping data and popularity", expanded=True):
    st.header("Understand the shopping data")
    st.write("In this section, we count how many receipts contain each product. A product scanned twice on one receipt still counts once. The denominator is all shopping baskets, so popularity is not the number of individual units sold.")
    csv_data = pd.read_csv(csv_path)
    sizes = csv_data.drop_duplicates(["TransactionID","Item"]).groupby("TransactionID").size()
    u,v = st.columns(2)
    u.metric("Average products per basket", f"{sizes.mean():.2f}")
    v.metric("Product rows in CSV", f"{len(csv_data):,}")
    u,v = st.columns(2)
    with u:
        st.subheader("10 most popular products")
        bar_chart(frequency_frame.head(10),"Product","Baskets (%)","Share of all baskets (%)")
        st.write("Reusable Bag appears in most baskets. This makes it easy for rules suggesting the bag to have high confidence even when the relationship is weak.")
    with v:
        st.subheader("5 least popular products")
        bar_chart(frequency_frame.tail(5),"Product","Baskets (%)","Share of all baskets (%)", "#397cad")
        st.write("Rare products have little evidence. At the saved support cutoff, a product present in fewer than 2% of receipts cannot participate in a frequent itemset.")
    st.subheader("How many products are in a basket?")
    distribution = sizes.value_counts().sort_index().rename_axis("Products per basket").reset_index(name="Receipts")
    chart = alt.Chart(distribution).mark_bar(color="#1769e0").encode(
        x=alt.X("Products per basket:O",title="Distinct products per receipt"),
        y=alt.Y("Receipts:Q"),tooltip=["Products per basket:O","Receipts:Q"])
    st.altair_chart(chart,width="stretch")
    st.caption("The bar height counts receipts of that size. Basket size ignores product quantities.")
    with st.expander("View the full popularity table"):
        st.dataframe(frequency_frame,hide_index=True,width="stretch")

with tab_learn, st.expander("Compare confidence and lift", expanded=False):
    st.header("Confidence versus lift")
    st.write("In this section, we compare simple rules: one starting product and one suggested product. This keeps each lift score tied to an individual product. The charts use all saved simple rules, independent of the basket sidebar filters.")
    simple = rule_frame[(rule_frame["Antecedent size"]==1)&(rule_frame["Consequent size"]==1)].copy()
    if simple.empty:
        st.info("No simple rules are present in this rules file.")
    else:
        simple["Group"] = simple["Suggested products"].apply(lambda x:"Suggests Reusable Bag" if x=="Reusable Bag" else "Other product")
        dots = alt.Chart(simple).mark_circle(size=85,opacity=0.8).encode(
            x=alt.X("Confidence (%):Q",scale=alt.Scale(domain=[0,100])),
            y=alt.Y("Lift:Q",scale=alt.Scale(zero=True)),
            color=alt.Color("Group:N",scale=alt.Scale(domain=["Suggests Reusable Bag","Other product"],range=["#e38318","#1769e0"])),
            tooltip=["Starting products","Suggested products",alt.Tooltip("Confidence (%):Q",format=".2f"),alt.Tooltip("Support (%):Q",format=".2f"),alt.Tooltip("Lift:Q",format=".3f")]
        ).properties(height=350)
        st.altair_chart(dots,width="stretch")
        st.write("Each dot is a rule. Moving right means higher confidence; moving up means higher lift. Orange dots suggest Reusable Bag: high confidence can coexist with lift near 1 because the product is already common. Hover over a dot to inspect the rule.")
        st.caption("Only exported rules with lift > 1 are available. This plot does not include baseline rules removed by that filter.")
        left,right=st.columns(2)
        for col,metric,title in [(left,"Confidence (%)","Top 10 by confidence"),(right,"Lift","Top 10 by lift")]:
            with col:
                st.subheader(title)
                top=simple.sort_values([metric,"Support (%)"],ascending=False).head(10).copy()
                top["Rule"] = top["Starting products"]+" → "+top["Suggested products"]
                bar_chart(top,"Rule",metric,metric)
        st.write("The confidence ranking rewards likely co-purchases, but the lift ranking compares that likelihood with overall popularity. Neither score alone proves a recommendation will increase sales.")
        chosen=st.selectbox("Explain a simple rule", range(len(simple)),format_func=lambda i: simple.iloc[i]["Starting products"]+" → "+simple.iloc[i]["Suggested products"],key="explain_simple")
        row=simple.iloc[chosen]
        baseline=popularity[row["Suggested products"]]
        st.info(f"For {row['Starting products']} → {row['Suggested products']}: confidence is {row['Confidence (%)']:.2f}%, compared with the suggested product's overall popularity of {baseline:.2%}. Lift is {row['Lift']:.3f}. About {round(row['Support (%)']/100*transaction_count):,} receipts contain both products.")
    st.subheader("Which products can the saved rules recommend?")
    recommended=set().union(*(r["consequents"] for r in rules)) if rules else set()
    covered=len(recommended.intersection(catalogue))
    st.metric("Catalogue coverage",f"{covered}/{len(catalogue)}",f"{covered/len(catalogue):.1%}",delta_color="off")
    st.write("Coverage counts products that can appear as suggestions in the full saved rule set, including multi-product consequents. It is not the percentage of visitors receiving a suggestion. Choosing single-product mode or stronger filters can reduce this coverage.")
    st.caption("Not covered: "+", ".join(sorted(set(catalogue)-recommended)))

with tab_learn, st.expander("How support changes the rules", expanded=False):
    st.header("What changes when support changes?")
    st.write("In this section, we review the offline Apriori experiment from the homework. Confidence stayed at 0.30 and itemset length was unrestricted. We do not re-mine rules when a visitor opens this page.")
    experiment=pd.DataFrame({"Minimum support":[0.001,0.01,0.02,0.10,0.20],
        "Frequent itemsets":[76341,3239,1072,100,6],"Rules produced":[456827,10851,2357,123,0]})
    st.caption("Recorded notebook results for the supplied seed-404 synthetic data (3,000 baskets, 44 products). These are historical experiment counts, not live recomputed results or runtime measurements.")
    if metadata.get("seed")!=404 or transaction_count!=3000 or len(catalogue)!=44:
        st.warning("The current data metadata differs from the original experiment. Treat these charts as an example, not results for the current dataset.")
    a,b=st.columns(2)
    for col,measure in [(a,"Frequent itemsets"),(b,"Rules produced")]:
        with col:
            chart=alt.Chart(experiment).mark_bar(color="#1769e0").encode(
                x=alt.X("Minimum support:O",sort=[0.001,0.01,0.02,0.1,0.2],title="Minimum support"),
                y=alt.Y(measure+":Q",scale=alt.Scale(type="symlog"),title=measure+" (compressed scale)"),
                tooltip=["Minimum support",measure]).properties(height=300)
            st.altair_chart(chart,width="stretch")
    st.write("The vertical axes use a compressed symlog scale so small counts remain visible beside very large ones; the bars are not on a linear scale. At support 0.001 we admit patterns present in only three baskets and generate many candidates. At support 0.20 we require 600 baskets, and no rules survive the confidence cutoff. Support 0.02 requires 60 baskets and is the saved baseline.")
    st.dataframe(experiment,hide_index=True,width="stretch")
    st.info("The sidebar support slider filters saved rules. Lowering it cannot recover rules that were never exported. Generating a new rule set requires an offline Apriori run and a new JSON export.")

with tab_learn, st.expander("Step-by-step guide", expanded=False):
    st.header("Follow the recommendation process")
    st.markdown("""1. **Generate and prepare data:** the notebook groups product rows into receipts and creates a boolean transaction table.
2. **Mine offline:** Apriori finds frequent combinations; association rules adds support, confidence, and lift.
3. **Save:** the notebook exports rules with support ≥ 0.02, confidence ≥ 0.30, and lift > 1.
4. **Match online:** we require every antecedent product in the selected basket, remove products already selected, and rank suggestions by lift.
5. **Apply:** a button adds products to the basket. The rules are looked up again; no model is retrained.
6. **Fallback:** when no association matches, the app displays clearly labelled popular products.""")
    st.subheader("Three scores, three different questions")
    st.dataframe(pd.DataFrame([
        {"Score":"Support","Question":"How common is the whole combination among all receipts?"},
        {"Score":"Confidence","Question":"Among receipts with the starting products, how often are the suggested products also present?"},
        {"Score":"Lift","Question":"How much more common is the suggested product here compared with its overall popularity?"}
    ]),hide_index=True,width="stretch")
    st.subheader("Try these examples")
    st.write("Charcoal: inspect barbecue suggestions. Diapers: inspect baby-product suggestions. Pasta and Garlic: try a rule with multiple starting products. Truffle Oil or an empty basket: check the popularity fallback. Apply one rule and confirm the added product is no longer suggested.")
    st.subheader("What this demo cannot establish")
    st.write("The data is synthetic, associations are not causal, and high lift does not guarantee useful recommendations. Homework mode assigns bundle lift to each product in a multi-product consequent; single-product mode avoids that specific issue. New and rare products may need category-based recommendations. Validate with future real receipts and customer testing before a business deployment.")
    st.caption("v1 is an educational demonstration. Graphs explain the saved data and rules; they do not measure real sales improvement.")


st.markdown("""
<style>
.stApp {background:#f8fafc !important; color:#172b4d !important;}
section[data-testid="stSidebar"] {background:#e8eef7 !important;}

div[data-testid="stButton"] button,
div[data-testid="stDownloadButton"] button {
    background:#174ea6 !important;
    border:2px solid #174ea6 !important;
    color:white !important;
    min-height:48px;
    border-radius:10px;
}
div[data-testid="stButton"] button *,
div[data-testid="stDownloadButton"] button * {
    color:white !important;
    font-weight:700 !important;
}
div[data-testid="stButton"] button:hover,
div[data-testid="stDownloadButton"] button:hover {
    background:#103675 !important;
    border-color:#103675 !important;
}
div[data-testid="stButton"] button:focus-visible {
    outline:3px solid #f59e0b !important;
    outline-offset:3px;
}
</style>
""", unsafe_allow_html=True)
# END CLEAR BUTTON STYLE
