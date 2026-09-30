"""Run with: python -m streamlit run app.py"""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Laurentienne | Apply grocery rules", page_icon="🛒", layout="wide")
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

st.markdown('''<div class="hero">
<div class="tag">ASSOCIATION RULES · APRIORI</div>
<h1>Grocery Recommendation System</h1>
<p style="color:#ffffff !important; font-size:1.2rem;">
<strong>Created by Abdelhafid Masmi</strong>
</p>
<p>Select products, discover recommendations, and apply matching rules.</p>
</div>''', unsafe_allow_html=True)
st.caption("Synthetic grocery data · Association rules · Abdelhafid Masmi")
a,b,c = st.columns(3)
a.metric("Shopping baskets", f"{transaction_count:,}")
b.metric("Products", len(catalogue))
c.metric("Saved rules", f"{len(rules):,}")

with st.sidebar:
    st.title("🛒 Basket settings")
    top_n = st.slider("Number of suggestions", 1, 8, 3)
    mode = st.radio("Recommendation mode", ["Single-product suggestions", "Homework rules"], help="Homework mode reproduces the notebook. Single-product mode evaluates rules with one suggested product.")
    min_lift = st.slider("Minimum lift (exclusive)", 1.0, 5.0, 1.0, 0.1)
    min_confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.30, 0.05)
    min_support = st.slider("Minimum support", 0.02, 0.20, 0.02, 0.01)
    st.divider()
    st.markdown("**How it works**")
    st.write("The app looks up your saved rules. It does not run Apriori while you shop.")
    st.caption("Higher lift indicates a stronger association relative to overall product popularity. It does not establish causation.")

st.subheader("1. Choose what the customer already has")
if "basket" not in st.session_state:
    st.session_state.basket = []

def set_basket(items):
    st.session_state.pop("last_applied", None)
    st.session_state.basket = [i for i in items if i in catalogue]

examples = st.columns(5)
for col,label,items in zip(examples,["🔥 Barbecue","👶 Baby essentials","🍝 Pasta night","☕ Breakfast","Clear basket"],[["Charcoal"],["Diapers"],["Pasta","Garlic"],["Coffee"],[]]):
    col.button(label,on_click=set_basket,args=(items,),width="stretch")
basket = st.multiselect("Products in your basket", catalogue, key="basket", placeholder="Choose one or more products…")
st.caption(f"{len(basket)} product(s) selected. Suggestions exclude products already in your basket.")

# Restrict the loaded rules without mining again.
active_rules = [r for r in rules if r["confidence"] >= min_confidence and r["support"] >= min_support]
matching = [r for r in active_rules if r["lift"] > min_lift
            and r["antecedents"].issubset(set(basket))
            and bool(r["consequents"] - set(basket))
            and (mode != "Single-product suggestions" or len(r["consequents"]) == 1)]
matching.sort(key=lambda r:(-r["lift"], -r["confidence"], sorted(r["antecedents"]), sorted(r["consequents"])))

st.subheader("2. Select and apply a matching rule")
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
results = recommend(basket,active_rules,top_n,min_lift,mode == "Single-product suggestions")
st.subheader("3. Recommended additions")
if results:
    st.write("Add one suggestion using its button, or add the full recommended list.")
    st.button("Add all recommended products", type="primary", on_click=set_basket,
              args=(basket+[r["item"] for r in results],), width="stretch")
    for start in range(0,len(results),3):
        cols = st.columns(3)
        for col,r in zip(cols,results[start:start+3]):
            with col:
                with st.container(border=True):
                    st.subheader(r["item"])
                    st.caption("Pairs with " + ", ".join(r["because"]))
                    st.metric("Rule lift", f"{r['lift']:.2f}×")
                    st.write(f"Confidence: **{r['confidence']:.1%}** · Support: **{r['support']:.1%}**")
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
    for start in range(0,len(fallback),3):
        cols = st.columns(3)
        for col,item in zip(cols,fallback[start:start+3]):
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
