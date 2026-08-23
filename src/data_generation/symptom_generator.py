import random
import json
from pathlib import Path

random.seed(42)

OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "synthetic" / "symptoms_5k.jsonl"

# 100 per condition: 15 EMERGENCY + 18 URGENT + 18 ROUTINE = 5,100 total
# Combined with real datasets (MedMCQA, MedDialog, medical-triage-500) → ~8-10K training samples
COUNTS = {"EMERGENCY": 100, "URGENT": 100, "ROUTINE": 100}

def c(*args):
    return random.choice(args)

def ri(lo, hi):
    return random.randint(lo, hi)

def rf(lo, hi, dp=1):
    return round(random.uniform(lo, hi), dp)

def sex():
    return c("male", "female")

def hx(*opts):
    return c(*opts)

# ── EMERGENCY generators (15 conditions) ─────────────────────────────────────

def gen_acute_mi():
    a = ri(45, 82)
    return (
        f"A {a}-year-old {sex()} presents with sudden {c('crushing','squeezing','pressure-like','tight','heavy')} "
        f"chest pain {c('radiating to the left arm','radiating to the jaw','radiating to both arms','with left shoulder pain','radiating to the neck')}, "
        f"{c('diaphoresis','profuse sweating','cold sweats')}, and {c('nausea','vomiting','shortness of breath','dizziness')} "
        f"for {ri(15,90)} minutes. {hx('History of hypertension.','Known hypertension and type 2 diabetes.','History of hyperlipidaemia.','No significant past medical history.','Prior MI 3 years ago.')}"
    )

def gen_stroke():
    a = ri(50, 85)
    side = c("left", "right")
    opp = "right" if side == "left" else "left"
    return (
        f"A {a}-year-old {sex()} with sudden onset {opp}-sided {c('weakness and facial drooping','arm and leg weakness','hemiplegia')}, "
        f"{c('slurred speech','aphasia','difficulty speaking','garbled speech')}, and "
        f"{c('gait instability','facial droop','visual disturbance','diplopia')} for {ri(20,120)} minutes. "
        f"{hx('History of atrial fibrillation.','Known hypertension.','History of hypertension and hyperlipidaemia.','On anticoagulation for AF.')}"
    )

def gen_anaphylaxis():
    a = ri(18, 65)
    trigger = c("bee sting","peanut ingestion","shellfish","penicillin","latex exposure","unknown allergen")
    return (
        f"A {a}-year-old {sex()} presenting with acute anaphylaxis following {trigger}: "
        f"{c('throat swelling','stridor','tongue angioedema')} with "
        f"{c('generalised urticaria','widespread hives','flushing')}, "
        f"BP {ri(60,80)}/{ri(30,50)} mmHg, "
        f"and {c('severe dyspnoea','bronchospasm','respiratory distress')}. "
        f"{hx('No prior allergy history.','Known allergy history — EpiPen at home, not used.','Previous mild reaction to NSAIDs.')}"
    )

def gen_respiratory_failure():
    a = ri(40, 80)
    o2 = ri(72, 84)
    return (
        f"A {a}-year-old {sex()} with acute {c('severe','profound')} shortness of breath, "
        f"oxygen saturation {o2}% on room air, "
        f"{c('coughing up blood','haemoptysis','blood-tinged sputum')}, "
        f"respiratory rate {ri(28,40)} breaths/min, and {c('cyanosis','central cyanosis','perioral cyanosis')}. "
        f"{hx('Background of COPD.','History of asthma.','No known respiratory disease.','Ex-smoker, 30 pack-year history.')}"
    )

def gen_meningitis():
    a = ri(1, 25)
    unit = "month-old" if a < 2 else "year-old"
    return (
        f"A {a}-{unit} {sex()} with high fever ({rf(39.5,41.0)}°C), "
        f"{c('non-blanching petechial rash','purpuric non-blanching rash','petechiae on trunk and limbs')}, "
        f"neck stiffness, {c('photophobia','phonophobia','photophobia and phonophobia')}, "
        f"and {c('vomiting','altered consciousness','lethargy')}. "
        f"Onset over {ri(4,24)} hours. {hx('No significant past history.','Unvaccinated child.','Up to date with vaccinations.')}"
    )

def gen_dka():
    a = ri(16, 65)
    glucose = ri(380, 650)
    return (
        f"A {a}-year-old {sex()} with known type {c('1','2')} diabetes presenting with blood glucose {glucose} mg/dL, "
        f"{c('confusion','altered mental status','drowsiness','GCS 12')}, "
        f"{c('fruity breath','acetone breath')}, "
        f"Kussmaul breathing, and {c('vomiting for 12 hours','nausea and vomiting since yesterday','polyuria and polydipsia for 3 days')}. "
        f"{hx('Non-compliant with insulin.','Recent viral illness.','No prior DKA episodes.')}"
    )

def gen_tension_pneumothorax():
    a = ri(18, 55)
    return (
        f"A {a}-year-old {sex()} following {c('blunt chest trauma in RTA','chest trauma from a fall','penetrating chest injury')}: "
        f"absent breath sounds on the {c('right','left')} side, "
        f"tracheal deviation, BP {ri(60,85)}/{ri(40,55)} mmHg, "
        f"HR {ri(120,145)} bpm, "
        f"and severe respiratory distress. "
        f"{hx('No past medical history.','History of previous spontaneous pneumothorax.')}"
    )

def gen_septic_shock():
    a = ri(50, 85)
    source = c("urinary tract","pulmonary","intra-abdominal","unknown")
    return (
        f"A {a}-year-old {sex()} with suspected {source} sepsis: "
        f"temperature {c(rf(38.5,40.0),'35.8','35.5')}°C, "
        f"HR {ri(115,145)} bpm, "
        f"BP {ri(65,85)}/{ri(35,50)} mmHg despite {ri(1,3)}L IV fluid, "
        f"GCS {ri(10,13)}, and lactate {rf(3.5,8.0)} mmol/L. "
        f"{hx('Immunocompromised on chemotherapy.','History of recurrent UTIs.','Recent abdominal surgery.','No significant past history.')}"
    )

def gen_status_epilepticus():
    a = ri(5, 60)
    duration = ri(6, 25)
    return (
        f"A {a}-year-old {sex()} with {c('tonic-clonic','generalised','focal')} seizure lasting {duration} minutes, "
        f"not self-terminating, {c('first episode ever','known epileptic on medication','previously febrile seizure at age 2')}. "
        f"Currently {c('post-ictal and unresponsive','still seizing on arrival','intermittently convulsing')}. "
        f"O2 sat {ri(82,91)}%, HR {ri(115,140)} bpm. "
        f"{hx('No prior seizure history.','Epilepsy diagnosed 3 years ago.','Head injury 6 months ago.')}"
    )

def gen_massive_pe():
    a = ri(35, 78)
    return (
        f"A {a}-year-old {sex()} with sudden onset {c('severe','profound')} dyspnoea and {c('pleuritic chest pain','central chest pain','right-sided chest pain')}, "
        f"HR {ri(118,145)} bpm, BP {ri(70,90)}/{ri(40,55)} mmHg, "
        f"O2 sat {ri(78,88)}% on room air, "
        f"and {c('right heart strain on ECG','raised troponin and BNP','elevated D-dimer > 5000')}. "
        f"{hx('Recent long-haul flight 3 days ago.','Post-op day 5 following knee replacement.','Known thrombophilia.','On OCP.')}"
    )

def gen_aortic_dissection():
    a = ri(50, 78)
    return (
        f"A {a}-year-old {sex()} with sudden onset {c('tearing','ripping','sharp')} chest pain "
        f"radiating to the {c('back','interscapular region','abdomen')}, "
        f"BP differential {ri(20,40)} mmHg between arms, "
        f"HR {ri(100,130)} bpm, diaphoretic. "
        f"{c('Pulses absent in right arm.','Absent left radial pulse.','Equal but weak peripheral pulses.')} "
        f"{hx('Long-standing hypertension.','Known Marfan syndrome.','History of bicuspid aortic valve.')}"
    )

def gen_ruptured_ectopic():
    a = ri(20, 38)
    return (
        f"A {a}-year-old female with sudden severe {c('left-sided','right-sided')} lower abdominal pain, "
        f"BP {ri(70,88)}/{ri(40,55)} mmHg, HR {ri(118,145)} bpm, "
        f"{c('positive pregnancy test','LMP 7 weeks ago','known intrauterine pregnancy not confirmed by USS')}, "
        f"peritonism on palpation, and {c('shoulder tip pain','referred shoulder pain','diaphragmatic irritation')}. "
        f"{hx('Previous ectopic pregnancy.','History of PID.','No significant gynaecological history.')}"
    )

def gen_polytrauma():
    a = ri(18, 55)
    mechanism = c("high-speed RTA","fall from height greater than 5 metres","industrial crush injury","ejection from vehicle")
    return (
        f"A {a}-year-old {sex()} following {mechanism}: "
        f"GCS {ri(6,11)}, BP {ri(70,90)}/{ri(40,55)} mmHg, HR {ri(120,145)} bpm, "
        f"suspected {c('pelvic fracture','femoral shaft fracture','haemopneumothorax')}, "
        f"and {c('scalp laceration with active bleeding','facial fractures','open tibial fracture')}. "
        f"Airway {c('compromised with blood','partially maintained','requiring adjunct')}."
    )

def gen_acute_limb_ischemia():
    a = ri(55, 80)
    limb = c("left leg","right leg","left arm")
    return (
        f"A {a}-year-old {sex()} with sudden onset {c('severe','agonising','extreme')} pain in the {limb}, "
        f"absent {c('popliteal and pedal','femoral','radial')} pulses, "
        f"{c('pallor','mottling','cyanosis')} and paraesthesia. "
        f"Onset {ri(1,4)} hours ago. "
        f"{hx('Known AF on warfarin — INR subtherapeutic.','Recent cardiac catheterisation.','History of peripheral arterial disease.')}"
    )

def gen_hypertensive_emergency():
    a = ri(45, 78)
    bp_s = ri(210, 260)
    bp_d = ri(120, 145)
    complication = c(
        "hypertensive encephalopathy with confusion and papilloedema",
        "acute pulmonary oedema and oxygen saturation 88%",
        "haematuria and AKI with creatinine 450 umol/L",
        "new onset focal neurological deficit",
    )
    return (
        f"A {a}-year-old {sex()} with BP {bp_s}/{bp_d} mmHg and {complication}. "
        f"{hx('Known hypertension, non-compliant with medications.','Hypertension on triple therapy.','Newly diagnosed hypertension.')}"
    )

EMERGENCY_GENERATORS = [
    gen_acute_mi, gen_stroke, gen_anaphylaxis, gen_respiratory_failure,
    gen_meningitis, gen_dka, gen_tension_pneumothorax, gen_septic_shock,
    gen_status_epilepticus, gen_massive_pe, gen_aortic_dissection,
    gen_ruptured_ectopic, gen_polytrauma, gen_acute_limb_ischemia,
    gen_hypertensive_emergency,
]

# ── URGENT generators (18 conditions) ────────────────────────────────────────

def gen_appendicitis():
    a = ri(12, 55)
    return (
        f"A {a}-year-old {sex()} with {ri(8,24)}-hour history of {c('right lower quadrant','periumbilical migrating to RLQ')} pain, "
        f"fever {rf(37.8,38.9)}°C, nausea{c(', and vomiting',', anorexia, and vomiting','')}. "
        f"Rebound tenderness present. WBC {rf(11.0,17.0)} × 10⁹/L. "
        f"{hx('No previous abdominal surgery.','Previous appendicitis ruled out 2 years ago.','No significant past history.')}"
    )

def gen_renal_colic():
    a = ri(20, 60)
    side = c("left","right")
    return (
        f"A {a}-year-old {sex()} with sudden onset severe {side}-sided loin-to-groin pain, "
        f"colicky in nature, pain score {ri(8,10)}/10, "
        f"{c('haematuria on dipstick','visible haematuria','microscopic haematuria')}, "
        f"and {c('vomiting','nausea and restlessness','diaphoresis')}. No fever. "
        f"{hx('Previous renal calculi.','No prior episodes.','Family history of kidney stones.')}"
    )

def gen_pneumonia():
    a = ri(25, 75)
    return (
        f"A {a}-year-old {sex()} with {ri(2,7)}-day history of productive cough with "
        f"{c('yellow','green','rusty')} sputum, fever {rf(38.5,40.2)}°C, "
        f"pleuritic chest pain, and O2 sat {ri(91,95)}% on room air. "
        f"Crackles on {c('right','left','bilateral')} auscultation. "
        f"{hx('Ex-smoker.','Current smoker.','No respiratory history.','Asplenic patient.')}"
    )

def gen_pyelonephritis():
    a = ri(18, 65)
    return (
        f"A {a}-year-old {sex()} with fever {rf(38.6,40.0)}°C, rigors, "
        f"{c('right','left','bilateral')} loin pain and tenderness, "
        f"dysuria and frequency for {ri(2,5)} days, "
        f"and {c('nausea','vomiting','lethargy')}. "
        f"Urine dipstick: leucocytes ++, nitrites positive. "
        f"{hx('Recurrent UTIs.','Previous pyelonephritis.','No urological history.','Pregnant at 16 weeks.')}"
    )

def gen_dvt():
    a = ri(28, 72)
    leg = c("left","right")
    return (
        f"A {a}-year-old {sex()} with {ri(2,7)}-day history of "
        f"{c('painful swollen','tender swollen','red swollen')} {leg} calf and ankle, "
        f"pitting oedema, and skin warmth. No dyspnoea currently. "
        f"Wells score {ri(2,4)}. "
        f"{hx('Recent long-haul flight.','Post-operative day 8 following hip replacement.','On OCP.','Known thrombophilia.','No provoking factors identified.')}"
    )

def gen_acute_glaucoma():
    a = ri(50, 78)
    eye = c("left","right")
    return (
        f"A {a}-year-old {sex()} with sudden severe {eye} eye pain, "
        f"blurred vision, halos around lights, and headache. "
        f"Eye appears red with corneal clouding. "
        f"IOP {ri(40,65)} mmHg on tonometry. "
        f"{c('Nausea and vomiting.','Associated vomiting.','Mild nausea.')} "
        f"{hx('No known ocular history.','History of narrow-angle glaucoma in other eye.','Hypermetropic patient.')}"
    )

def gen_threatened_miscarriage():
    a = ri(20, 38)
    weeks = ri(6, 19)
    return (
        f"A {a}-year-old female at {weeks} weeks gestation with "
        f"{c('light vaginal bleeding','moderate vaginal bleeding','vaginal spotting')} and "
        f"{c('mild lower abdominal cramping','intermittent pelvic pain','lower abdominal discomfort')}. "
        f"Cervical os {c('closed on speculum examination','closed on bimanual','not yet assessed')}. "
        f"{hx('No previous miscarriages.','One previous miscarriage at 8 weeks.','IVF pregnancy.')}"
    )

def gen_acute_heart_failure():
    a = ri(60, 85)
    return (
        f"A {a}-year-old {sex()} with {ri(2,5)}-day worsening shortness of breath, "
        f"orthopnoea, {c('bilateral','right-sided','left-sided')} ankle oedema, "
        f"O2 sat {ri(89,94)}% on room air, "
        f"and {c('bibasal crackles','pulmonary crackles to mid-zones','bilateral pleural effusions on CXR')}. "
        f"BP {ri(150,190)}/{ri(90,110)} mmHg. "
        f"{hx('Known ischaemic cardiomyopathy.','History of hypertension and AF.','Previous CABG.')}"
    )

def gen_dental_abscess():
    a = ri(20, 60)
    return (
        f"A {a}-year-old {sex()} with {ri(3,10)}-day history of severe dental pain, "
        f"progressive {c('facial swelling','submandibular swelling','cheek swelling')} now involving the {c('floor of mouth','neck','parapharyngeal space')}, "
        f"fever {rf(37.9,39.5)}°C, "
        f"and {c('odynophagia','difficulty swallowing','trismus')} limiting oral intake. "
        f"{hx('Poor dentition.','Previous dental abscess.','No recent dental treatment.')}"
    )

def gen_moderate_tbi():
    a = ri(18, 65)
    mechanism = c("fall from standing","assault","cycling accident without helmet","RTA — airbag deployed")
    return (
        f"A {a}-year-old {sex()} following {mechanism}: "
        f"GCS {ri(12,14)}, {c('one episode of vomiting','two episodes of vomiting','persistent vomiting')}, "
        f"{c('brief loss of consciousness','LOC for approximately 2 minutes','no LOC but post-traumatic amnesia')}, "
        f"and {c('scalp laceration','frontal haematoma','periorbital bruising')}. "
        f"Currently {c('confused','disorientated to time','orientated but drowsy')}."
    )

def gen_cellulitis():
    a = ri(25, 75)
    limb = c("right lower leg","left lower leg","right forearm","face")
    return (
        f"A {a}-year-old {sex()} with {ri(3,7)}-day spreading {c('erythema','redness')} of the {limb}, "
        f"warmth, tenderness, and marked oedema, "
        f"with red margin {c('advancing rapidly','extending 3 cm in 24 hours','spreading despite oral antibiotics')}. "
        f"Temperature {rf(37.9,39.2)}°C. "
        f"{hx('Type 2 diabetes.','Chronic venous insufficiency.','Previous cellulitis requiring IV antibiotics.','Lymphoedema.')}"
    )

def gen_urinary_retention():
    a = ri(55, 82)
    return (
        f"A {a}-year-old {sex()} with {ri(8,24)}-hour inability to void, "
        f"suprapubic pain and distension, bladder volume {ri(600,900)} mL on USS. "
        f"In significant {c('discomfort','distress','pain')}. "
        f"{hx('Known benign prostatic hyperplasia.','Previous urinary retention.','On anticholinergic medication.','No prior urological problems.')}"
    )

def gen_stable_ectopic():
    a = ri(22, 38)
    return (
        f"A {a}-year-old female with {ri(6,9)} weeks amenorrhoea, "
        f"positive pregnancy test, {c('left-sided','right-sided')} pelvic pain, "
        f"and {c('light vaginal bleeding','brown vaginal discharge','minimal spotting')}. "
        f"Haemodynamically stable: BP {ri(105,125)}/{ri(65,80)} mmHg, HR {ri(78,98)} bpm. "
        f"{hx('Previous ectopic pregnancy.','History of PID.','IVF conception.','No gynaecological history.')}"
    )

def gen_severe_migraine():
    a = ri(18, 52)
    return (
        f"A {a}-year-old {sex()} with {c('known migraine','known migrainous headaches','migraine since teenage years')}: "
        f"severe {c('unilateral','left-sided','right-sided','holocranial')} throbbing headache, "
        f"pain {ri(8,10)}/10, "
        f"photophobia, phonophobia, {c('vomiting × 4 episodes','persistent vomiting','unable to keep fluids down')}, "
        f"lasting {ri(18,48)} hours and not responding to home analgesia. "
        f"{hx('On topiramate prophylaxis.','No regular preventative therapy.','Previous ED attendances for migraine.')}"
    )

def gen_bowel_obstruction():
    a = ri(45, 80)
    return (
        f"A {a}-year-old {sex()} with {ri(24,72)}-hour history of colicky abdominal pain, "
        f"absolute constipation, abdominal distension, and {c('multiple episodes of vomiting','bilious vomiting','faeculent vomiting')}. "
        f"Bowel sounds {c('tinkling','high-pitched','absent')}. Abdomen soft, no peritonism. "
        f"{hx('Previous laparotomy.','No abdominal surgery.','Known colorectal cancer on surveillance.')}"
    )

def gen_tia():
    a = ri(55, 82)
    return (
        f"A {a}-year-old {sex()} with {ri(30,180)}-minute episode of "
        f"{c('right arm weakness and facial droop','left-sided weakness','slurred speech and left hand weakness','visual loss in right eye')} "
        f"that fully resolved {ri(1,8)} hours ago. Currently neurologically intact. "
        f"ABCD2 score {ri(4,6)}. "
        f"{hx('Known hypertension and hyperlipidaemia.','History of AF.','Previous TIA 2 years ago.','No prior neurological events.')}"
    )

def gen_moderate_hyperglycemia():
    a = ri(30, 75)
    glucose = ri(280, 420)
    return (
        f"A {a}-year-old {sex()} with type {c('1','2')} diabetes presenting with blood glucose {glucose} mg/dL, "
        f"{c('polyuria and polydipsia for 3 days','lethargy and malaise','nausea without vomiting')}, "
        f"no ketonuria, "
        f"and {c('mild dehydration','dry mucous membranes','reduced skin turgor')}. "
        f"BP stable. No confusion. "
        f"{hx('Non-compliant with insulin.','Recent corticosteroid course.','Vomiting illness preventing oral medication.')}"
    )

def gen_laceration():
    a = ri(18, 70)
    site = c("hand","forearm","scalp","lower leg","face")
    return (
        f"A {a}-year-old {sex()} with a {c('deep','gaping','full-thickness')} {site} laceration "
        f"from {c('kitchen knife injury','broken glass','machinery','fall')}. "
        f"Bleeding {c('controlled with direct pressure','oozing but controlled','not actively bleeding')}. "
        f"Wound {ri(3,10)} cm, requires suturing. Neurovascular status intact. "
        f"{hx('Up to date with tetanus.','Tetanus status unknown.','Last tetanus booster > 10 years ago.')}"
    )

URGENT_GENERATORS = [
    gen_appendicitis, gen_renal_colic, gen_pneumonia, gen_pyelonephritis,
    gen_dvt, gen_acute_glaucoma, gen_threatened_miscarriage, gen_acute_heart_failure,
    gen_dental_abscess, gen_moderate_tbi, gen_cellulitis, gen_urinary_retention,
    gen_stable_ectopic, gen_severe_migraine, gen_bowel_obstruction,
    gen_tia, gen_moderate_hyperglycemia, gen_laceration,
]

# ── ROUTINE generators (18 conditions) ───────────────────────────────────────

def gen_urti():
    a = ri(18, 65)
    return (
        f"A {a}-year-old {sex()} with {ri(2,5)}-day history of "
        f"{c('sore throat','mild pharyngitis','scratchy throat')}, "
        f"{c('rhinorrhoea','nasal congestion','runny nose')}, "
        f"{c('low-grade fever','temperature 37.5°C','afebrile')}, and "
        f"{c('mild myalgia','fatigue','mild cough')}. "
        f"No dysphagia, no stridor, no rash. Self-managing with paracetamol. "
        f"{hx('No significant history.','Mild asthma — well controlled.','Non-smoker.')}"
    )

def gen_back_pain():
    a = ri(22, 60)
    return (
        f"A {a}-year-old {sex()} with {c('lower','lumbar')} back pain following "
        f"{c('heavy lifting','prolonged sitting at a desk','awkward movement','gardening')} "
        f"{ri(1,7)} days ago. Pain {ri(3,6)}/10, {c('no radiation to legs','no neurological symptoms','no bladder or bowel symptoms')}. "
        f"Able to weight-bear. "
        f"{hx('No previous episodes.','Recurrent mechanical back pain.','Sedentary occupation.')}"
    )

def gen_tension_headache():
    a = ri(20, 55)
    return (
        f"A {a}-year-old {sex()} with {c('bilateral','diffuse','band-like')} headache, "
        f"pain {ri(3,6)}/10, present for {ri(1,5)} days, "
        f"responding partially to {c('paracetamol','ibuprofen','over-the-counter analgesia')}. "
        f"No vomiting, no photophobia, no neck stiffness. {c('Associated with work stress.','After prolonged screen use.','No clear trigger.')} "
        f"{hx('Recurrent tension headaches.','No prior headache history.','Migraines excluded by neurologist.')}"
    )

def gen_ankle_sprain():
    a = ri(16, 45)
    ankle = c("left","right")
    return (
        f"A {a}-year-old {sex()} with {ankle} ankle sprain sustained during "
        f"{c('football','running','basketball','stepping off a kerb')} {ri(1,3)} days ago. "
        f"Mild swelling and bruising over the lateral malleolus. "
        f"Able to weight-bear, pain {ri(2,5)}/10. Ottawa rules negative. "
        f"{hx('No prior ankle injuries.','Previous sprain same ankle.','No bony tenderness.')}"
    )

def gen_dm_review():
    a = ri(40, 75)
    hba1c = rf(6.5, 9.5)
    return (
        f"A {a}-year-old {sex()} with type 2 diabetes attending for routine HbA1c review. "
        f"Last HbA1c {hba1c}%. "
        f"Blood pressure {ri(120,145)}/{ri(75,90)} mmHg. "
        f"No hypoglycaemic episodes. "
        f"{c('On metformin and sitagliptin.','On metformin only.','On insulin glargine and metformin.')} "
        f"Foot exam and eye screening up to date. "
        f"{hx('No diabetic complications.','Background diabetic retinopathy.','Microalbuminuria.')}"
    )

def gen_eczema():
    a = ri(18, 50)
    site = c("forearms","hands","antecubital fossae","neck","face")
    return (
        f"A {a}-year-old {sex()} with known eczema presenting with a flare on the {site}. "
        f"Itching, erythema, and dry skin. "
        f"No signs of secondary infection. "
        f"Triggered by {c('soap change','stress','cold weather','unknown')}. "
        f"{hx('Managed with emollients and mild topical steroids.','On dietary restriction for food allergy.','Long history of atopic eczema.')}"
    )

def gen_lower_uti():
    a = ri(18, 65)
    return (
        f"A {a}-year-old {sex()} with {ri(2,4)}-day history of dysuria, urinary frequency, and urgency. "
        f"No fever, no loin pain, no systemic symptoms. "
        f"Urine dipstick: leucocytes +, nitrites positive. "
        f"{hx('Recurrent UTIs.','First episode.','Post-coital UTIs — requesting prophylaxis review.')}"
    )

def gen_prescription_refill():
    a = ri(40, 80)
    meds = c(
        "antihypertensives (amlodipine and ramipril)",
        "levothyroxine for hypothyroidism",
        "salbutamol and beclomethasone inhalers",
        "atorvastatin and aspirin",
        "metformin and linagliptin",
    )
    return (
        f"A {a}-year-old {sex()} attending for routine prescription refill of {meds}. "
        f"Condition stable and well controlled. "
        f"No new symptoms. BP {ri(118,138)}/{ri(72,88)} mmHg today. "
        f"{hx('Long-term condition, well managed.','Recent blood tests normal.','Adherent to medication.')}"
    )

def gen_vaccination():
    a = ri(18, 80)
    vaccine = c(
        "annual influenza vaccination",
        "COVID-19 booster",
        "pneumococcal vaccination",
        "shingles (zoster) vaccination",
        "travel vaccines (hepatitis A and typhoid)",
    )
    return (
        f"A {a}-year-old {sex()} attending for {vaccine}. "
        f"No acute complaints. "
        f"Medically {c('fit and well','stable on regular medications','no significant past history')}. "
        f"No contraindications to vaccination."
    )

def gen_knee_oa():
    a = ri(55, 80)
    knee = c("bilateral","right","left")
    return (
        f"A {a}-year-old {sex()} with {knee} knee osteoarthritis attending for pain management review. "
        f"Pain {ri(3,6)}/10 on mobilising, worse on stairs. "
        f"No joint effusion currently. "
        f"Managing with {c('paracetamol and topical diclofenac','physiotherapy and NSAIDs','paracetamol alone')}. "
        f"{hx('X-ray confirmed moderate OA.','Awaiting orthopaedic outpatient review.','Not yet referred to orthopaedics.')}"
    )

def gen_gastroenteritis():
    a = ri(18, 60)
    return (
        f"A {a}-year-old {sex()} with {ri(1,3)}-day history of "
        f"{c('watery diarrhoea', 'loose stools', f'diarrhoea x {ri(3,8)} episodes per day')} and "
        f"{c('mild nausea','nausea and one episode of vomiting','nausea without vomiting')}. "
        f"No blood in stool. Tolerating oral fluids. "
        f"Afebrile. No systemic symptoms. "
        f"{hx('Recent takeaway meal.','Possible contact with unwell family member.','No recent travel.')}"
    )

def gen_conjunctivitis():
    a = ri(18, 60)
    eye = c("bilateral","right","left")
    return (
        f"A {a}-year-old {sex()} with {eye} eye redness, "
        f"{c('watery discharge','purulent discharge','sticky discharge on waking')}, "
        f"and mild irritation for {ri(2,5)} days. "
        f"Visual acuity normal. No photophobia. No corneal involvement. "
        f"{hx('No prior eye problems.','Seasonal allergies.','Contact lens wearer.')}"
    )

def gen_insect_bite():
    a = ri(18, 65)
    site = c("left forearm","right hand","lower leg","neck","back")
    return (
        f"A {a}-year-old {sex()} with an insect bite on the {site} from {ri(1,5)} days ago. "
        f"Localised erythema {ri(2,5)} cm diameter, mild itching, no systemic symptoms. "
        f"No spreading redness. Afebrile. No lymphadenopathy. "
        f"{hx('No known allergies.','History of mild local reactions.','First bite reaction.')}"
    )

def gen_minor_wound():
    a = ri(18, 65)
    return (
        f"A {a}-year-old {sex()} with a minor {c('abrasion','superficial laceration','graze')} on the "
        f"{c('knee','palm','shin','elbow')} from a {c('fall','cycling incident','garden injury')}. "
        f"Wound clean, {ri(1,3)} cm, not requiring sutures. "
        f"Bleeding stopped. Neurovascular status intact. "
        f"{hx('Tetanus up to date.','Tetanus booster given 5 years ago.','Tetanus status to be checked.')}"
    )

def gen_otitis_media():
    a = ri(1, 12)
    unit = "year-old" if a > 1 else "month-old"
    return (
        f"A {a}-{unit} {sex()} with {ri(2,5)}-day history of ear pain, "
        f"{c('pulling at the ear','irritability','disrupted sleep')}, and "
        f"{c('low-grade fever','temperature 37.8°C','afebrile')}. "
        f"Tympanic membrane {c('erythematous and bulging','inflamed','dull with loss of light reflex')} on otoscopy. "
        f"{hx('Previous episodes of otitis media.','First episode.','Completed antibiotics 3 months ago for AOM.')}"
    )

def gen_hypertension_review():
    a = ri(45, 78)
    bp = f"{ri(128,150)}/{ri(78,95)}"
    return (
        f"A {a}-year-old {sex()} with known hypertension attending for routine blood pressure review. "
        f"BP today {bp} mmHg. "
        f"No headache, no visual changes, no chest pain. "
        f"{c('On amlodipine 5 mg.','On ramipril 10 mg and amlodipine 5 mg.','On losartan 50 mg.')} "
        f"Recent renal function and electrolytes normal. "
        f"{hx('Hypertension for 8 years.','Diagnosed hypertension 2 years ago.','Strong family history of hypertension.')}"
    )

def gen_constipation():
    a = ri(25, 80)
    return (
        f"A {a}-year-old {sex()} with {ri(5,14)}-day history of constipation, "
        f"last bowel movement {ri(5,14)} days ago. "
        f"Mild bloating and discomfort but no vomiting, no rectal bleeding. "
        f"Tolerating oral intake. Abdomen soft, mild lower abdominal fullness. "
        f"{hx('Known IBS-C.','On regular opioid analgesia.','No previous bowel problems.','Low fibre diet.')}"
    )

def gen_skin_lesion():
    a = ri(30, 75)
    lesion = c("pigmented mole","sebaceous cyst","skin tag","wart on finger","actinic keratosis on forearm")
    return (
        f"A {a}-year-old {sex()} requesting review of a {lesion} "
        f"that has been present for {ri(3,24)} months. "
        f"No recent change in size or colour. No bleeding. No associated symptoms. "
        f"{hx('Fair skin, significant sun exposure history.','No family history of melanoma.','Previous benign lesions removed.')}"
    )

ROUTINE_GENERATORS = [
    gen_urti, gen_back_pain, gen_tension_headache, gen_ankle_sprain,
    gen_dm_review, gen_eczema, gen_lower_uti, gen_prescription_refill,
    gen_vaccination, gen_knee_oa, gen_gastroenteritis, gen_conjunctivitis,
    gen_insect_bite, gen_minor_wound, gen_otitis_media,
    gen_hypertension_review, gen_constipation, gen_skin_lesion,
]

# ── Main generation ───────────────────────────────────────────────────────────

def generate_all():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    schedule = [
        ("EMERGENCY", EMERGENCY_GENERATORS, COUNTS["EMERGENCY"]),
        ("URGENT",    URGENT_GENERATORS,    COUNTS["URGENT"]),
        ("ROUTINE",   ROUTINE_GENERATORS,   COUNTS["ROUTINE"]),
    ]

    total = 0
    with open(OUTPUT_PATH, "w") as f:
        for label, generators, per_condition in schedule:
            class_total = 0
            for gen_fn in generators:
                for _ in range(per_condition):
                    desc = gen_fn()
                    f.write(json.dumps({"symptom_description": desc, "triage_level": label}) + "\n")
                    class_total += 1
            print(f"{label}: {class_total} samples")
            total += class_total

    print(f"\nTotal: {total} samples saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_all()
