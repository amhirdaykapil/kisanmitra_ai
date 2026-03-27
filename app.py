import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from datetime import datetime

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="KisanDrishti AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════
# CLASS NAMES
# After Colab training finishes, paste the printed CLASS_NAMES
# from Step 7 here to replace this list.
# ══════════════════════════════════════════════════════════════
CLASS_NAMES = [
    "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
    "Blueberry___healthy","Cherry_(including_sour)___Powdery_mildew","Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight","Corn_(maize)___healthy","Grape___Black_rot",
    "Grape___Esca_(Black_Measles)","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot","Peach___healthy",
    "Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy","Potato___Early_blight",
    "Potato___Late_blight","Potato___healthy","Raspberry___healthy","Soybean___healthy",
    "Squash___Powdery_mildew","Strawberry___Leaf_scorch","Strawberry___healthy",
    "Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight","Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot","Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus","Tomato___healthy"
]

# ══════════════════════════════════════════════════════════════
# TRANSLATIONS
# ══════════════════════════════════════════════════════════════
LANG = {
    "en": {
        "hero_tag":"🌾 Precision Agriculture AI",
        "hero_title":"Detect Crop Disease Instantly",
        "hero_desc":"Upload a leaf photo and get AI-powered disease diagnosis in seconds. Supports major crops across 38 disease conditions — built for real Indian farmers.",
        "stat_conditions":"Conditions","stat_crops":"Crops","stat_images":"Images","stat_model":"Model",
        "upload_title":"Upload Leaf Image",
        "upload_hint":"JPG, PNG · Drag & drop · Close-up shots work best",
        "analyze_btn":"🔍 Analyze Leaf",
        "analyzing":"Analyzing leaf...",
        "detected_crop":"DETECTED CROP",
        "confidence":"Confidence","severity":"Severity",
        "weather_title":"🌦️ Weather Risk Alert",
        "top_preds":"Top Predictions",
        "treatment_title":"Recommended Treatment",
        "history_title":"Scan History",
        "no_history":"No scans yet — upload a leaf!",
        "tips_title":"💡 Tips for Best Results",
        "tip1":"📸 Take photos in natural daylight",
        "tip2":"🍃 Fill the frame with the leaf",
        "tip3":"🔁 Scan multiple leaves for accuracy",
        "tip4":"🌿 Avoid wet or dirty leaves",
        "low_conf_warn":"⚠️ Low confidence — try a clearer, closer photo",
        "risk_high":"⚠️ HIGH disease risk today",
        "risk_med":"🟡 MODERATE disease risk",
        "risk_low":"✅ LOW disease risk today",
        "weather_detail":"Humidity {}% · Wind {} km/h",
        "location_fail":"Allow location for weather data",
        "footer":"🌿 KisanDrishti AI — Built for Bharat's Farmers | Hackathon 2026 | MobileNetV2",
    },
    "hi": {
        "hero_tag":"🌾 स्मार्ट कृषि AI",
        "hero_title":"फसल की बीमारी तुरंत पहचानें",
        "hero_desc":"पत्ती की फोटो अपलोड करें और सेकंडों में AI से बीमारी जानें। 38 बीमारियाँ — भारतीय किसानों के लिए।",
        "stat_conditions":"बीमारियाँ","stat_crops":"फसलें","stat_images":"तस्वीरें","stat_model":"मॉडल",
        "upload_title":"पत्ती की फोटो अपलोड करें",
        "upload_hint":"JPG, PNG · खींचें या क्लिक करें · नज़दीकी फोटो सबसे अच्छी",
        "analyze_btn":"🔍 जाँच करें",
        "analyzing":"पत्ती की जाँच हो रही है...",
        "detected_crop":"पहचानी गई फसल",
        "confidence":"विश्वसनीयता","severity":"गंभीरता",
        "weather_title":"🌦️ मौसम जोखिम चेतावनी",
        "top_preds":"शीर्ष अनुमान",
        "treatment_title":"सुझाया गया उपचार",
        "history_title":"स्कैन इतिहास",
        "no_history":"अभी कोई स्कैन नहीं — पत्ती अपलोड करें!",
        "tips_title":"💡 बेहतर परिणाम के लिए सुझाव",
        "tip1":"📸 प्राकृतिक रोशनी में फोटो लें",
        "tip2":"🍃 पत्ती से पूरा फ्रेम भरें",
        "tip3":"🔁 कई पत्तियाँ स्कैन करें",
        "tip4":"🌿 गीली या गंदी पत्ती से बचें",
        "low_conf_warn":"⚠️ कम विश्वसनीयता — बेहतर फोटो लें",
        "risk_high":"⚠️ आज बीमारी का खतरा अधिक है",
        "risk_med":"🟡 मध्यम बीमारी जोखिम",
        "risk_low":"✅ आज बीमारी का खतरा कम है",
        "weather_detail":"आर्द्रता {}% · हवा {} km/h",
        "location_fail":"मौसम के लिए स्थान की अनुमति दें",
        "footer":"🌿 किसानदृष्टि AI — भारत के किसानों के लिए | हैकाथॉन 2026 | MobileNetV2",
    }
}

TREATMENTS = {
    "en":{
        "Apple___Apple_scab":{"name":"Fungicide Spray","text":"Apply captan or myclobutanil fungicide. Rake and destroy fallen leaves.","steps":["Spray at bud break","Repeat every 7–10 days in wet weather","Remove infected fruit promptly"]},
        "Apple___Black_rot":{"name":"Prune & Spray","text":"Remove infected wood and mummified fruit. Apply thiophanate-methyl fungicide.","steps":["Cut 8–10 inches below infection","Disinfect pruning tools","Apply copper spray after pruning"]},
        "Apple___Cedar_apple_rust":{"name":"Preventive Fungicide","text":"Apply myclobutanil at pink bud stage. Remove nearby juniper hosts.","steps":["Spray at first orange spores","Repeat every 7 days","Remove infected leaves"]},
        "Apple___healthy":{"name":"✅ Healthy!","text":"Apple crop looks great. Continue regular scouting and orchard hygiene.","steps":["Scout weekly","Maintain pruning","Monitor soil nutrients"]},
        "Blueberry___healthy":{"name":"✅ Healthy!","text":"Blueberry plants healthy. Maintain soil acidity and drainage.","steps":["Check pH 4.5–5.5","Mulch to retain moisture","Prune dead canes"]},
        "Cherry_(including_sour)___Powdery_mildew":{"name":"Sulfur Spray","text":"Apply wettable sulfur. Improve air circulation by pruning.","steps":["Spray at first white powder","Avoid overhead irrigation","Remove infected shoots"]},
        "Cherry_(including_sour)___healthy":{"name":"✅ Healthy!","text":"Cherry crop healthy. Monitor pests.","steps":["Scout for fruit fly","Maintain soil moisture","Prune after harvest"]},
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":{"name":"Strobilurin Fungicide","text":"Apply azoxystrobin. Practice crop rotation and remove crop debris.","steps":["Apply at first lesions","Rotate crops every 2 years","Plow residues after harvest"]},
        "Corn_(maize)___Common_rust_":{"name":"Triazole Fungicide","text":"Apply propiconazole or tebuconazole. Plant resistant hybrids next season.","steps":["Spray at early rust signs","Avoid excess nitrogen","Monitor humidity"]},
        "Corn_(maize)___Northern_Leaf_Blight":{"name":"Mancozeb + Propiconazole","text":"Apply at first lesion. Plant resistant varieties. Manage crop residue.","steps":["Apply at silking stage","Till residues post-harvest","Use certified disease-free seeds"]},
        "Corn_(maize)___healthy":{"name":"✅ Healthy!","text":"Corn crop healthy. Maintain soil nutrition.","steps":["Monitor for armyworm","Ensure adequate nitrogen","Scout weekly"]},
        "Grape___Black_rot":{"name":"Mancozeb Spray","text":"Apply mancozeb. Remove mummified berries and infected canes.","steps":["Spray from budbreak to veraison","Remove infected clusters","Improve canopy airflow"]},
        "Grape___Esca_(Black_Measles)":{"name":"Prune & Protect","text":"No chemical cure. Remove infected wood, protect cuts with wound sealant.","steps":["Remove infected cordons","Seal pruning wounds immediately","Avoid vine stress"]},
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":{"name":"Copper Fungicide","text":"Apply copper oxychloride. Maintain good vineyard hygiene.","steps":["Spray after rain events","Remove fallen infected leaves","Maintain row spacing"]},
        "Grape___healthy":{"name":"✅ Healthy!","text":"Grapevine healthy. Maintain canopy management.","steps":["Prune annually","Monitor for mealybugs","Test soil nutrients"]},
        "Orange___Haunglongbing_(Citrus_greening)":{"name":"⚠️ No Cure — Act Fast","text":"Remove infected trees immediately. Control psyllid vectors with insecticide.","steps":["Remove & destroy infected trees","Apply imidacloprid for psyllids","Plant certified nursery stock","Report to authorities"]},
        "Peach___Bacterial_spot":{"name":"Copper Bactericide","text":"Apply copper hydroxide sprays. Use resistant varieties.","steps":["Spray at petal fall","Repeat every 10–14 days","Avoid wounding fruit"]},
        "Peach___healthy":{"name":"✅ Healthy!","text":"Peach crop healthy. Maintain irrigation.","steps":["Thin fruit for sizing","Watch for leaf curl","Apply dormant oil in winter"]},
        "Pepper,_bell___Bacterial_spot":{"name":"Copper + Mancozeb","text":"Apply copper-based bactericide. Avoid working with wet plants.","steps":["Spray every 5–7 days in wet weather","Remove infected leaves","Use drip irrigation"]},
        "Pepper,_bell___healthy":{"name":"✅ Healthy!","text":"Bell pepper healthy. Keep up watering schedule.","steps":["Stake plants","Monitor for aphids","Apply balanced NPK"]},
        "Potato___Early_blight":{"name":"Chlorothalonil Spray","text":"Apply chlorothalonil at first sign. Improve drainage.","steps":["Remove infected lower leaves","Spray every 7–10 days","Mulch soil to prevent splash"]},
        "Potato___Late_blight":{"name":"🚨 Urgent — Metalaxyl","text":"Extremely destructive! Apply metalaxyl IMMEDIATELY. Remove infected plants.","steps":["Remove & burn infected plants NOW","Apply metalaxyl systemically","Stop overhead irrigation","Alert neighboring farms"]},
        "Potato___healthy":{"name":"✅ Healthy!","text":"Potato crop healthy. Monitor during humid periods.","steps":["Hill soil around plants","Scout every 3–4 days in monsoon","Ensure good drainage"]},
        "Raspberry___healthy":{"name":"✅ Healthy!","text":"Raspberry canes healthy.","steps":["Remove old canes after fruiting","Mulch around base","Fertilize in early spring"]},
        "Soybean___healthy":{"name":"✅ Healthy!","text":"Soybean crop healthy.","steps":["Check nitrogen nodules","Scout for pod borers","Maintain soil moisture"]},
        "Squash___Powdery_mildew":{"name":"Potassium Bicarbonate","text":"Apply potassium bicarbonate or neem oil. Improve spacing.","steps":["Spray undersides of leaves","Remove infected leaves","Avoid excess nitrogen"]},
        "Strawberry___Leaf_scorch":{"name":"Captan Fungicide","text":"Apply captan. Remove infected leaves and improve drainage.","steps":["Remove & destroy infected leaves","Avoid overhead watering","Renovate bed after harvest"]},
        "Strawberry___healthy":{"name":"✅ Healthy!","text":"Strawberry plants healthy.","steps":["Thin runners","Side-dress with fertilizer","Monitor for slugs"]},
        "Tomato___Bacterial_spot":{"name":"Copper Bactericide","text":"Apply copper hydroxide. Use disease-free transplants.","steps":["Spray every 7 days","Remove infected leaves","Use drip irrigation"]},
        "Tomato___Early_blight":{"name":"Mancozeb Spray","text":"Apply mancozeb. Remove lower infected leaves.","steps":["Remove infected lower leaves first","Spray every 7–10 days","Stake plants for airflow"]},
        "Tomato___Late_blight":{"name":"🚨 Urgent — Systemic Fungicide","text":"Apply metalaxyl immediately. Highly contagious!","steps":["Remove infected plants urgently","Apply metalaxyl or cymoxanil","Stop overhead watering","Alert neighboring growers"]},
        "Tomato___Leaf_Mold":{"name":"Ventilation + Fungicide","text":"Improve ventilation. Apply mancozeb.","steps":["Reduce humidity below 85%","Remove infected leaves","Spray undersides"]},
        "Tomato___Septoria_leaf_spot":{"name":"Chlorothalonil Spray","text":"Apply chlorothalonil at first sign.","steps":["Begin spray at transplanting","Remove infected leaves weekly","Avoid overhead irrigation"]},
        "Tomato___Spider_mites Two-spotted_spider_mite":{"name":"Miticide Spray","text":"Apply abamectin. Increase humidity.","steps":["Spray undersides of leaves","Repeat every 5–7 days","Avoid broad-spectrum pesticides"]},
        "Tomato___Target_Spot":{"name":"Azoxystrobin Spray","text":"Apply azoxystrobin. Improve airflow.","steps":["Spray at first lesion signs","Stake and prune for airflow","Remove infected material"]},
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus":{"name":"Vector Control","text":"No cure. Control whiteflies with imidacloprid immediately.","steps":["Remove infected plants","Apply imidacloprid for whiteflies","Use reflective mulch","Plant resistant varieties"]},
        "Tomato___Tomato_mosaic_virus":{"name":"Remove & Prevent","text":"No cure. Remove infected plants. Control aphid vectors.","steps":["Remove & destroy infected plants","Sanitize tools with bleach","Control aphids with neem oil","Use resistant seeds"]},
        "Tomato___healthy":{"name":"✅ Healthy!","text":"Tomato plants are healthy!","steps":["Prune suckers","Fertilize with potassium at fruiting","Monitor weekly"]},
    },
    "hi":{
        "Corn_(maize)___Common_rust_":{"name":"ट्राइएज़ोल फफूंदनाशी","text":"प्रोपिकोनाज़ोल या टेबुकोनाज़ोल लगाएं।","steps":["जंग के लक्षणों पर स्प्रे करें","अत्यधिक नाइट्रोजन से बचें","नमी की निगरानी करें"]},
        "Corn_(maize)___Northern_Leaf_Blight":{"name":"मैन्कोज़ेब + प्रोपिकोनाज़ोल","text":"पहले धब्बे पर लगाएं।","steps":["सिल्किंग चरण में स्प्रे करें","फसल अवशेष जलाएं","प्रमाणित बीज उपयोग करें"]},
        "Corn_(maize)___healthy":{"name":"✅ स्वस्थ!","text":"मक्के की फसल स्वस्थ है।","steps":["साप्ताहिक निगरानी करें","नाइट्रोजन सुनिश्चित करें"]},
        "Potato___Early_blight":{"name":"क्लोरोथैलोनिल स्प्रे","text":"मैन्कोज़ेब लगाएं। जल निकासी सुधारें।","steps":["संक्रमित पत्तियाँ हटाएं","7-10 दिन में स्प्रे करें","मिट्टी पलवार करें"]},
        "Potato___Late_blight":{"name":"🚨 तुरंत — मेटालेक्सिल","text":"बेहद खतरनाक! तुरंत मेटालेक्सिल लगाएं।","steps":["संक्रमित पौधे तुरंत जलाएं","मेटालेक्सिल लगाएं","ऊपरी सिंचाई बंद करें","पड़ोसी किसानों को सूचित करें"]},
        "Potato___healthy":{"name":"✅ स्वस्थ!","text":"आलू की फसल स्वस्थ है।","steps":["पौधों के पास मिट्टी चढ़ाएं","हर 3-4 दिन जाँचें"]},
        "Tomato___Early_blight":{"name":"मैन्कोज़ेब स्प्रे","text":"मैन्कोज़ेब लगाएं। निचली पत्तियाँ हटाएं।","steps":["संक्रमित पत्तियाँ हटाएं","7-10 दिन में स्प्रे करें","पौधे को सहारा दें"]},
        "Tomato___Late_blight":{"name":"🚨 तुरंत — सिस्टमिक फफूंदनाशी","text":"मेटालेक्सिल तुरंत लगाएं।","steps":["संक्रमित पौधे हटाएं","मेटालेक्सिल लगाएं","ऊपरी सिंचाई बंद करें"]},
        "Tomato___healthy":{"name":"✅ स्वस्थ!","text":"टमाटर के पौधे स्वस्थ हैं!","steps":["साइड शूट हटाएं","फलते समय पोटेशियम डालें"]},
        "Apple___healthy":{"name":"✅ स्वस्थ!","text":"सेब की फसल स्वस्थ है।","steps":["साप्ताहिक निगरानी","कटाई-छंटाई बनाए रखें"]},
    }
}

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
if "lang" not in st.session_state:      st.session_state.lang = "en"
if "scan_history" not in st.session_state: st.session_state.scan_history = []
if "weather" not in st.session_state:  st.session_state.weather = None

def t(key):
    l = st.session_state.lang or "en"
    return LANG[l].get(key, LANG["en"].get(key, key))

# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Mukta:wght@300;400;600;700&display=swap');
:root{--green:#3dba6f;--green2:#56e08a;--gold:#e8c84a;--red:#e05555;--cream:#f0ead8;--muted:rgba(240,234,216,0.55);--card:rgba(255,255,255,0.04);--border:rgba(61,186,111,0.2);}
html,body,[class*="css"]{font-family:'Mukta',sans-serif!important;background:#0a1f0d!important;color:#f0ead8!important;}
.main .block-container{padding:1.5rem 2rem!important;max-width:1300px!important;}
.stApp{background:radial-gradient(ellipse 60% 50% at 15% 15%,rgba(61,186,111,0.10) 0%,transparent 60%),radial-gradient(ellipse 50% 60% at 85% 85%,rgba(232,200,74,0.05) 0%,transparent 60%),linear-gradient(160deg,#0a1f0d 0%,#0d2812 50%,#0a1f0d 100%)!important;}
.hero-tag{display:inline-block;background:rgba(61,186,111,0.1);border:1px solid rgba(61,186,111,0.3);border-radius:20px;padding:.25rem .9rem;font-size:.72rem;font-weight:700;letter-spacing:2px;color:#3dba6f;text-transform:uppercase;margin-bottom:.8rem;}
.hero-title{font-family:'Playfair Display',serif!important;font-size:2.5rem!important;font-weight:900!important;line-height:1.1!important;color:#fff!important;margin-bottom:.7rem!important;}
.hero-desc{color:rgba(240,234,216,0.55);font-size:.95rem;line-height:1.7;margin-bottom:1.5rem;}
.stat-row{display:flex;gap:.8rem;flex-wrap:wrap;}
.stat-card{flex:1;min-width:70px;background:rgba(61,186,111,0.08);border:1px solid rgba(61,186,111,0.2);border-radius:12px;padding:.7rem;text-align:center;}
.stat-num{font-family:'Playfair Display',serif;font-size:1.5rem;font-weight:700;color:#3dba6f;}
.stat-lbl{font-size:.68rem;color:rgba(240,234,216,0.55);text-transform:uppercase;letter-spacing:1px;}
.upload-hint-box{border:2px dashed rgba(61,186,111,0.35);border-radius:16px;padding:1.5rem;text-align:center;margin-bottom:1rem;}
.upload-hint-icon{font-size:2.5rem;margin-bottom:.5rem;}
.upload-hint-title{font-family:'Playfair Display',serif;font-size:1.1rem;color:#3dba6f;margin-bottom:.3rem;}
.upload-hint-sub{font-size:.8rem;color:rgba(240,234,216,0.55);}
.result-card{background:linear-gradient(135deg,rgba(22,61,28,0.85),rgba(15,46,20,0.6));border:1px solid rgba(61,186,111,0.3);border-radius:20px;padding:1.5rem;margin-bottom:1rem;}
.result-card.disease{background:linear-gradient(135deg,rgba(55,15,15,0.9),rgba(35,10,10,0.7));border-color:rgba(224,85,85,0.35);}
.r-tag{font-size:.68rem;letter-spacing:3px;text-transform:uppercase;color:#e8c84a;}
.r-crop{font-family:'Playfair Display',serif;font-size:2.3rem;font-weight:900;color:#fff;line-height:1;}
.r-disease{font-size:1.2rem;font-weight:600;margin-top:.2rem;}
.r-ok{color:#3dba6f;} .r-sick{color:#e05555;}
.conf-labels{display:flex;justify-content:space-between;font-size:.78rem;color:rgba(240,234,216,0.55);margin-bottom:.3rem;}
.conf-track{background:rgba(255,255,255,0.08);border-radius:999px;height:8px;overflow:hidden;}
.conf-fill{height:100%;border-radius:999px;}
.conf-high{background:linear-gradient(90deg,#2d8f52,#56e08a);}
.conf-mid{background:linear-gradient(90deg,#c47a00,#e8a020);}
.conf-low{background:linear-gradient(90deg,#8b1111,#e05555);}
.sev-row{display:flex;align-items:center;gap:.4rem;margin-top:.8rem;}
.sev-lbl{font-size:.7rem;color:rgba(240,234,216,0.55);text-transform:uppercase;letter-spacing:1px;margin-right:.3rem;}
.sev-block{height:7px;flex:1;border-radius:4px;background:rgba(255,255,255,0.08);}
.s-mild{background:#e8c84a!important;} .s-mod{background:#e07a22!important;} .s-sev{background:#e05555!important;}
.weather-card{background:linear-gradient(135deg,rgba(10,25,60,0.85),rgba(8,18,45,0.7));border:1px solid rgba(100,160,255,0.2);border-radius:20px;padding:1.5rem;}
.w-tag{font-size:.68rem;letter-spacing:3px;text-transform:uppercase;color:rgba(100,160,255,0.8);margin-bottom:.7rem;}
.w-temp{font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;color:#fff;}
.w-city{font-size:.82rem;color:rgba(240,234,216,0.55);margin-bottom:.5rem;}
.risk-badge{display:inline-block;border-radius:20px;padding:.3rem .9rem;font-size:.75rem;font-weight:700;margin-top:.5rem;}
.risk-high{background:rgba(224,85,85,0.2);color:#e05555;border:1px solid rgba(224,85,85,0.3);}
.risk-med{background:rgba(232,200,74,0.15);color:#e8c84a;border:1px solid rgba(232,200,74,0.3);}
.risk-low{background:rgba(61,186,111,0.15);color:#3dba6f;border:1px solid rgba(61,186,111,0.3);}
.pred-row{display:flex;align-items:center;gap:.7rem;padding:.5rem 0;border-bottom:1px solid rgba(255,255,255,0.05);}
.pred-row:last-child{border:none;}
.pred-rank{font-family:'Playfair Display',serif;font-size:1rem;font-weight:700;color:#e8c84a;width:22px;}
.pred-name{flex:1;font-size:.82rem;color:#f0ead8;}
.pred-bar-bg{width:55px;background:rgba(255,255,255,0.07);border-radius:999px;height:4px;overflow:hidden;}
.pred-bar-fill{height:100%;border-radius:999px;background:#3dba6f;}
.pred-pct{font-size:.82rem;font-weight:600;color:#3dba6f;width:40px;text-align:right;}
.section-title{font-family:'Playfair Display',serif;font-size:1rem;color:#f0ead8;margin-bottom:.8rem;}
.treat-card{background:var(--card);border:1px solid var(--border);border-radius:20px;padding:1.5rem;}
.treat-name{font-family:'Playfair Display',serif;font-size:1.1rem;color:#fff;margin-bottom:.4rem;}
.treat-text{font-size:.87rem;color:rgba(240,234,216,0.55);line-height:1.6;margin-bottom:.8rem;}
.treat-step{display:flex;align-items:flex-start;gap:.5rem;margin-bottom:.4rem;font-size:.83rem;color:rgba(240,234,216,0.6);}
.step-dot{width:6px;height:6px;border-radius:50%;background:#3dba6f;margin-top:6px;flex-shrink:0;}
.history-item{display:flex;align-items:center;gap:.7rem;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:.55rem .8rem;margin-bottom:.5rem;font-size:.82rem;}
.h-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.h-ok{background:#3dba6f;} .h-sick{background:#e05555;}
.h-name{flex:1;color:#f0ead8;} .h-conf{color:rgba(240,234,216,0.55);font-size:.78rem;} .h-time{color:rgba(240,234,216,0.3);font-size:.75rem;}
.tip-item{display:flex;gap:.6rem;padding:.5rem 0;border-bottom:1px solid rgba(255,255,255,0.05);font-size:.83rem;}
.tip-item:last-child{border:none;} .tip-text{color:rgba(240,234,216,0.6);line-height:1.5;}
.g-card{background:var(--card);border:1px solid var(--border);border-radius:20px;padding:1.5rem;margin-bottom:1rem;}
.kisan-nav{display:flex;align-items:center;justify-content:space-between;padding:.8rem 0;margin-bottom:1.5rem;border-bottom:1px solid var(--border);}
.kisan-logo{font-family:'Playfair Display',serif;font-size:1.5rem;font-weight:900;color:#fff;}
.kisan-logo span{color:#3dba6f;}
.kisan-badge{background:rgba(61,186,111,0.12);border:1px solid rgba(61,186,111,0.3);color:#3dba6f;border-radius:20px;padding:.2rem .8rem;font-size:.7rem;font-weight:700;letter-spacing:1px;}
.warn-box{background:rgba(232,200,74,0.08);border:1px solid rgba(232,200,74,0.3);border-radius:10px;padding:.8rem 1rem;font-size:.83rem;color:#e8c84a;margin-top:.8rem;}
div[data-testid="stFileUploadDropzone"]{background:rgba(61,186,111,0.05)!important;border:2px dashed rgba(61,186,111,0.35)!important;border-radius:12px!important;}
.stButton>button{background:linear-gradient(135deg,#3dba6f,#2a9e58)!important;color:white!important;border:none!important;border-radius:12px!important;padding:.7rem 1.5rem!important;font-family:'Mukta',sans-serif!important;font-size:1rem!important;font-weight:600!important;width:100%!important;}
.stButton>button:hover{box-shadow:0 10px 25px rgba(61,186,111,0.3)!important;}
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# LANGUAGE POPUP
# ══════════════════════════════════════════════════════════════
if st.session_state.lang is None:
    st.markdown("""
    <style>
    .lang-overlay{position:fixed;inset:0;z-index:9999;backdrop-filter:blur(24px);background:rgba(10,31,13,0.88);display:flex;align-items:center;justify-content:center;}
    .lang-box{background:linear-gradient(135deg,rgba(22,61,28,0.97),rgba(15,46,20,0.97));border:1px solid rgba(61,186,111,0.45);border-radius:24px;padding:3rem 3.5rem;text-align:center;max-width:440px;width:90%;box-shadow:0 40px 80px rgba(0,0,0,0.7);}
    .lang-leaf{font-size:3.5rem;display:block;}
    .lang-ptitle{font-family:'Playfair Display',serif;font-size:2.1rem;font-weight:900;color:#fff;margin:.5rem 0 .2rem;}
    .lang-psub{color:rgba(240,234,216,0.5);font-size:.9rem;margin-bottom:1.5rem;}
    </style>
    <div class="lang-overlay">
      <div class="lang-box">
        <span class="lang-leaf">🌿</span>
        <div class="lang-ptitle">KisanDrishti AI</div>
        <div class="lang-psub">किसान की आँख — The Farmer's Eye</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    _,mid,_ = st.columns([1,2,1])
    with mid:
        st.markdown("<div style='margin-top:18rem'></div>", unsafe_allow_html=True)
        if st.button("🇬🇧  Continue in English", key="le"):
            st.session_state.lang = "en"; st.rerun()
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        if st.button("🇮🇳  हिन्दी में जारी रखें", key="lh"):
            st.session_state.lang = "hi"; st.rerun()
    st.stop()

# ══════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras")

def preprocess(img):
    img = img.convert("RGB").resize((224,224))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, 0)

# ══════════════════════════════════════════════════════════════
# WEATHER
# ══════════════════════════════════════════════════════════════
def get_weather():
    try:
        ip = requests.get("https://ipapi.co/json/", timeout=4).json()
        lat,lon,city = ip.get("latitude",28.6),ip.get("longitude",77.2),ip.get("city","India")
        url=(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
             f"&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&timezone=auto")
        w=requests.get(url,timeout=5).json()["current"]
        code=w["weather_code"]
        icon="☀️" if code==0 else "⛅" if code<=3 else "🌫️" if code<=49 else "🌧️" if code<=69 else "🌨️" if code<=79 else "⛈️"
        return {"temp":round(w["temperature_2m"]),"hum":w["relative_humidity_2m"],
                "wind":round(w["wind_speed_10m"]),"city":city,"icon":icon,"ok":True}
    except:
        return {"ok":False}

# ══════════════════════════════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════════════════════════════
n1,n2 = st.columns([4,1])
with n1:
    st.markdown(f'<div class="kisan-nav"><div class="kisan-logo">🌿 Kisan<span>Drishti</span></div><span class="kisan-badge">AI POWERED · {len(CLASS_NAMES)} CONDITIONS</span></div>', unsafe_allow_html=True)
with n2:
    other = "हिन्दी" if st.session_state.lang=="en" else "English"
    if st.button(f"🌐 {other}"):
        st.session_state.lang = "hi" if st.session_state.lang=="en" else "en"
        st.rerun()

# ══════════════════════════════════════════════════════════════
# HERO + UPLOAD
# ══════════════════════════════════════════════════════════════
h1,h2 = st.columns([1.1,1], gap="large")

with h1:
    st.markdown(f"""
    <div class="hero-tag">{t('hero_tag')}</div>
    <div class="hero-title">{t('hero_title')}</div>
    <div class="hero-desc">{t('hero_desc')}</div>
    <div class="stat-row">
      <div class="stat-card"><div class="stat-num">38</div><div class="stat-lbl">{t('stat_conditions')}</div></div>
      <div class="stat-card"><div class="stat-num">10+</div><div class="stat-lbl">{t('stat_crops')}</div></div>
      <div class="stat-card"><div class="stat-num">87K</div><div class="stat-lbl">{t('stat_images')}</div></div>
      <div class="stat-card"><div class="stat-num">MNV2</div><div class="stat-lbl">{t('stat_model')}</div></div>
    </div>
    """, unsafe_allow_html=True)

with h2:
    st.markdown(f"""
    <div class="upload-hint-box">
      <div class="upload-hint-icon">📷</div>
      <div class="upload-hint-title">{t('upload_title')}</div>
      <div class="upload-hint-sub">{t('upload_hint')}</div>
    </div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
    clicked  = st.button(t("analyze_btn"))

# ══════════════════════════════════════════════════════════════
# PREVIEW ONLY
# ══════════════════════════════════════════════════════════════
if uploaded and not clicked:
    _,mid,_ = st.columns([1,2,1])
    with mid:
        st.image(Image.open(uploaded), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# ANALYSIS + RESULTS
# ══════════════════════════════════════════════════════════════
if uploaded and clicked:
    image = Image.open(uploaded)

    with st.spinner(t("analyzing")):
        try:
            model  = load_model()
            preds  = model.predict(preprocess(image), verbose=0)[0]
        except Exception as e:
            st.error(f"Model error: {e}")
            st.stop()

    idx        = int(np.argmax(preds))
    confidence = float(preds[idx]) * 100
    label      = CLASS_NAMES[idx]
    crop       = label.split("___")[0].replace("_"," ").replace("(","").replace(")","").replace(",","").strip()
    disease    = label.split("___")[1].replace("_"," ").strip()
    healthy    = "healthy" in disease.lower()

    # History
    st.session_state.scan_history.insert(0,{"crop":crop,"disease":disease,"conf":confidence,"healthy":healthy,"time":datetime.now().strftime("%H:%M")})
    if len(st.session_state.scan_history)>6: st.session_state.scan_history.pop()

    # Weather once
    if not st.session_state.weather:
        st.session_state.weather = get_weather()

    st.markdown("---")

    # ── ROW 1: Image | Result | Weather ──────────────────────
    c1,c2,c3 = st.columns([1,1.2,0.9], gap="medium")

    with c1:
        st.image(image, use_container_width=True)

    with c2:
        cc = " disease" if not healthy else ""
        dc = "r-ok" if healthy else "r-sick"
        di = "✅" if healthy else "⚠️"
        fc = "conf-high" if confidence>=70 else "conf-mid" if confidence>=50 else "conf-low"
        s1=s2=s3=""
        if not healthy:
            s1="s-mild"
            if confidence>=55: s2="s-mod"
            if confidence>=75: s3="s-sev"
        st.markdown(f"""
        <div class="result-card{cc}">
          <div class="r-tag">{t('detected_crop')}</div>
          <div class="r-crop">{crop}</div>
          <div class="r-disease {dc}">{di} {disease}</div>
          <div style="margin-top:1rem">
            <div class="conf-labels"><span>{t('confidence')}</span><span>{confidence:.1f}%</span></div>
            <div class="conf-track"><div class="conf-fill {fc}" style="width:{confidence:.1f}%"></div></div>
          </div>
          <div class="sev-row">
            <span class="sev-lbl">{t('severity')}</span>
            <div class="sev-block {s1}"></div>
            <div class="sev-block {s2}"></div>
            <div class="sev-block {s3}"></div>
          </div>
        </div>
        {"<div class='warn-box'>"+t('low_conf_warn')+"</div>" if confidence<60 and not healthy else ""}
        """, unsafe_allow_html=True)

    with c3:
        w = st.session_state.weather
        if w and w.get("ok"):
            hum=w["hum"]
            rc = "risk-high" if hum>80 else "risk-med" if hum>60 else "risk-low"
            rt = t("risk_high") if hum>80 else t("risk_med") if hum>60 else t("risk_low")
            wd = t("weather_detail").format(hum,w["wind"])
            st.markdown(f"""
            <div class="weather-card">
              <div class="w-tag">{t('weather_title')}</div>
              <div style="font-size:2rem;margin:.3rem 0">{w['icon']}</div>
              <div class="w-temp">{w['temp']}°C</div>
              <div class="w-city">{w['city']}</div>
              <div style="font-size:.8rem;color:rgba(240,234,216,0.5)">{wd}</div>
              <div><span class="risk-badge {rc}">{rt}</span></div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="weather-card">
              <div class="w-tag">{t('weather_title')}</div>
              <div style="font-size:2rem">🌿</div>
              <div class="w-temp">--°C</div>
              <div class="w-city" style="font-size:.85rem;color:rgba(240,234,216,0.4)">{t('location_fail')}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── ROW 2: Top Preds | Treatment | History ────────────────
    p1,p2,p3 = st.columns([1,1,1], gap="medium")

    with p1:
        top5 = np.argsort(preds)[-5:][::-1]
        rows=""
        for i,ix in enumerate(top5):
            nm=CLASS_NAMES[ix].replace("___"," — ").replace("_"," ")
            pc=preds[ix]*100
            rows+=f'<div class="pred-row"><span class="pred-rank">#{i+1}</span><span class="pred-name">{nm}</span><div class="pred-bar-bg"><div class="pred-bar-fill" style="width:{min(pc,100):.0f}%"></div></div><span class="pred-pct">{pc:.1f}%</span></div>'
        st.markdown(f'<div class="g-card"><div class="section-title">📊 {t("top_preds")}</div>{rows}</div>', unsafe_allow_html=True)

    with p2:
        db=TREATMENTS.get(st.session_state.lang,TREATMENTS["en"])
        tr=db.get(label) or TREATMENTS["en"].get(label) or {"name":"Consult Expert","text":"Contact your local agricultural officer.","steps":[]}
        sh="".join(f'<div class="treat-step"><div class="step-dot"></div><span>{s}</span></div>' for s in tr["steps"])
        st.markdown(f'<div class="treat-card"><div class="section-title">💊 {t("treatment_title")}</div><div class="treat-name">{tr["name"]}</div><div class="treat-text">{tr["text"]}</div>{sh}</div>', unsafe_allow_html=True)

    with p3:
        hist_rows=""
        for h in st.session_state.scan_history:
            dc="h-ok" if h["healthy"] else "h-sick"
            hist_rows+=f'<div class="history-item"><div class="h-dot {dc}"></div><span class="h-name">{h["crop"]} — {h["disease"]}</span><span class="h-conf">{h["conf"]:.0f}%</span><span class="h-time">{h["time"]}</span></div>'
        no_h=f'<div style="color:rgba(240,234,216,0.4);font-size:.85rem;padding:1rem 0">{t("no_history")}</div>'
        st.markdown(f'<div class="g-card"><div class="section-title">🕐 {t("history_title")}</div>{hist_rows or no_h}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TIPS (always visible)
# ══════════════════════════════════════════════════════════════
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
tc,_ = st.columns([2,1])
with tc:
    st.markdown(f"""
    <div class="g-card">
      <div class="section-title">{t('tips_title')}</div>
      <div class="tip-item"><span>📸</span><span class="tip-text">{t('tip1')}</span></div>
      <div class="tip-item"><span>🍃</span><span class="tip-text">{t('tip2')}</span></div>
      <div class="tip-item"><span>🔁</span><span class="tip-text">{t('tip3')}</span></div>
      <div class="tip-item"><span>🌿</span><span class="tip-text">{t('tip4')}</span></div>
    </div>""", unsafe_allow_html=True)

# FOOTER
st.markdown(f'<div style="text-align:center;padding:1.5rem;border-top:1px solid rgba(61,186,111,0.2);color:rgba(240,234,216,0.4);font-size:.8rem;margin-top:1.5rem">{t("footer")}</div>', unsafe_allow_html=True)
