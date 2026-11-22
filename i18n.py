"""English / Lao text for the Streamlit demo.

Short strings are looked up by their English text in ``LAO``; anything missing
falls back to English.  Longer markdown passages live in ``BLOCKS`` as
English/Lao pairs.  To correct a Lao wording, edit the value here only.
"""

from __future__ import annotations

LANGUAGES = {"lo": "ລາວ", "en": "English"}
DEFAULT_LANGUAGE = "lo"

LAO: dict[str, str] = {
    # Page and masthead
    "Presentation": "ການນຳສະເໜີ",
    "Demo": "ສາທິດ",
    "Calculating…": "ກຳລັງຄຳນວນ…",
    "Laos population forecast": "ການຄາດຄະເນປະຊາກອນລາວ",
    "By district, age and sex, from Ministry of Health DHIS2 data.": (
        "ແຍກຕາມເມືອງ, ອາຍຸ ແລະ ເພດ, ຈາກຂໍ້ມູນ DHIS2 ຂອງກະຊວງສາທາລະນະສຸກ."
    ),
    "Laos population: training, test and forecast years": (
        "ປະຊາກອນລາວ: ປີຝຶກສອນ, ປີທົດສອບ ແລະ ປີຄາດຄະເນ"
    ),
    "Train": "ຝຶກສອນ",
    "Test": "ທົດສອບ",
    "Forecast": "ຄາດຄະເນ",
    "Population in {year}": "ປະຊາກອນ ປີ {year}",
    "Test-year MAPE {mape}": "MAPE ປີທົດສອບ {mape}",
    "{density} people per km²": "{density} ຄົນ ຕໍ່ ກມ²",
    "{change} since {year}": "{change} ນັບແຕ່ປີ {year}",
    "Forecast for {year}": "ຄາດຄະເນປີ {year}",
    "{change} vs {year}": "{change} ທຽບກັບປີ {year}",
    "Density in {year} (people per km²)": "ຄວາມໜາແໜ້ນ ປີ {year} (ຄົນ ຕໍ່ ກມ²)",
    "{change} people per km² vs {year}": "{change} ຄົນ ຕໍ່ ກມ² ທຽບກັບປີ {year}",
    "Test-year MAPE": "MAPE ປີທົດສອບ",
    "Model used for the forecast: {model}.": "ແບບຈຳລອງທີ່ໃຊ້ຄາດຄະເນ: {model}.",
    "Mean absolute percentage error on the unseen {start}–{end} test years. The bars "
    "show the error in each test year.": (
        "ເປີເຊັນຄ່າຜິດພາດສຳບູນສະເລ່ຍ ໃນປີທົດສອບ {start}–{end} ທີ່ແບບຈຳລອງບໍ່ເຄີຍເຫັນ. "
        "ແທ່ງສະແດງຄ່າຜິດພາດຂອງແຕ່ລະປີທົດສອບ."
    ),
    "Show": "ສະແດງ",
    "Annual growth": "ອັດຕາເຕີບໂຕຕໍ່ປີ",
    "Forecast ({model})": "ຄາດຄະເນ ({model})",
    "Population trajectory {start}–{end}": "ແນວໂນ້ມປະຊາກອນ {start}–{end}",
    "Share": "ສັດສ່ວນ",
    "Age structure: {start} vs {end}": "ໂຄງສ້າງອາຍຸ: {start} ທຽບກັບ {end}",
    "Share of population (%)": "ສັດສ່ວນຂອງປະຊາກອນ (%)",
    "Average annual growth {start}–{end}: {rate}.": "ອັດຕາເຕີບໂຕສະເລ່ຍຕໍ່ປີ {start}–{end}: {rate}.",
    "Land area {area} km², measured from the map boundaries.": (
        "ເນື້ອທີ່ດິນ {area} ກມ², ວັດຈາກຂອບເຂດໃນແຜນທີ່."
    ),
    "{year}, {model}": "ປີ {year}, {model}",
    "{rows} data rows. Baseline: {baseline}. Source: DHIS2, Ministry of Health.": (
        "{rows} ແຖວຂໍ້ມູນ. ຂໍ້ມູນພື້ນຖານ: {baseline}. ແຫຼ່ງຂໍ້ມູນ: DHIS2, ກະຊວງສາທາລະນະສຸກ."
    ),
    # Sidebar
    "Data and forecast": "ຂໍ້ມູນ ແລະ ການຄາດຄະເນ",
    "Upload population CSV": "ອັບໂຫຼດໄຟລ໌ CSV ປະຊາກອນ",
    "Forecast through": "ຄາດຄະເນຮອດປີ",
    "Outlier correction threshold": "ເກນການແກ້ໄຂຄ່າຜິດປົກກະຕິ",
    "Lower values flag and potentially correct more observations. Higher values "
    "require more exceptional robust scores. Every correction still requires an "
    "independent demographic confirmation.": (
        "ຄ່າຕ່ຳຈະໝາຍ ແລະ ອາດແກ້ໄຂຂໍ້ມູນຫຼາຍຂຶ້ນ. ຄ່າສູງຕ້ອງການຄະແນນທົນທານ "
        "(robust score) ທີ່ຜິດປົກກະຕິຫຼາຍກວ່າ. ທຸກການແກ້ໄຂຍັງຕ້ອງໄດ້ຮັບການຢືນຢັນ "
        "ຈາກການກວດສອບດ້ານປະຊາກອນອີກວິທີໜຶ່ງ."
    ),
    "Forecast model": "ແບບຈຳລອງການຄາດຄະເນ",
    "All models": "ທຸກແບບຈຳລອງ",
    "Choose one model to inspect, or All models to compare their results and accuracy.": (
        "ເລືອກແບບຈຳລອງໜຶ່ງເພື່ອເບິ່ງລາຍລະອຽດ, ຫຼື ເລືອກທຸກແບບຈຳລອງ "
        "ເພື່ອປຽບທຽບຜົນ ແລະ ຄວາມຊັດເຈນ."
    ),
    "Districts": "ເມືອງ",
    "Sex": "ເພດ",
    "Age groups": "ກຸ່ມອາຍຸ",
    "All districts": "ທຸກເມືອງ",
    "All sexes": "ທຸກເພດ",
    "All age groups": "ທຸກກຸ່ມອາຍຸ",
    "Baseline data type": "ປະເພດຂໍ້ມູນພື້ນຖານ",
    "Original data": "ຂໍ້ມູນຕົ້ນສະບັບ",
    "Outlier-corrected data (threshold {threshold})": (
        "ຂໍ້ມູນທີ່ແກ້ໄຂຄ່າຜິດປົກກະຕິແລ້ວ (ເກນ {threshold})"
    ),
    "Choose the baseline used by the forecast. Outlier-corrected data changes only "
    "temporal anomalies independently confirmed by a demographic check; the original "
    "upload is never overwritten.": (
        "ເລືອກຂໍ້ມູນພື້ນຖານທີ່ໃຊ້ຄາດຄະເນ. ຂໍ້ມູນທີ່ແກ້ໄຂແລ້ວ ປ່ຽນສະເພາະຄ່າຜິດປົກກະຕິ"
        "ຕາມເວລາ ທີ່ໄດ້ຮັບການຢືນຢັນຈາກການກວດສອບດ້ານປະຊາກອນ; ໄຟລ໌ຕົ້ນສະບັບຈະບໍ່ຖືກຂຽນທັບ."
    ),
    "Default data file not found: {path}": "ບໍ່ພົບໄຟລ໌ຂໍ້ມູນເລີ່ມຕົ້ນ: {path}",
    "Could not load the population data: {error}": "ບໍ່ສາມາດໂຫຼດຂໍ້ມູນປະຊາກອນໄດ້: {error}",
    "The selected filters contain no records. Change or clear the filters.": (
        "ບໍ່ມີຂໍ້ມູນຕາມຕົວກອງທີ່ເລືອກ. ກະລຸນາປ່ຽນ ຫຼື ລ້າງຕົວກອງ."
    ),
    "5–7 minute demo path": "ລຳດັບການສາທິດ 5–7 ນາທີ",
    # Validation and tabs
    "Time-based validation model comparison": "ປຽບທຽບແບບຈຳລອງ ດ້ວຍການກວດສອບຕາມເວລາ",
    "Models are trained on 2016–2022 and evaluated on the unseen 2023–2025 holdout "
    "period. The final 2026+ forecasts are then retrained on all 2016–2025 data.": (
        "ແບບຈຳລອງຖືກຝຶກສອນດ້ວຍຂໍ້ມູນ 2016–2022 ແລະ ປະເມີນຜົນດ້ວຍຂໍ້ມູນ 2023–2025 "
        "ທີ່ແບບຈຳລອງບໍ່ເຄີຍເຫັນ. ຈາກນັ້ນຈຶ່ງຝຶກສອນຄືນໃໝ່ດ້ວຍຂໍ້ມູນທັງໝົດ 2016–2025 "
        "ເພື່ອຄາດຄະເນປີ 2026 ເປັນຕົ້ນໄປ."
    ),
    "Model": "ແບບຈຳລອງ",
    "Accuracy (%)": "ຄວາມຊັດເຈນ (%)",
    "Forecast Overview": "ພາບລວມການຄາດຄະເນ",
    "Source data & outliers": "ຂໍ້ມູນຕົ້ນສະບັບ ແລະ ຄ່າຜິດປົກກະຕິ",
    "Population Visualization": "ການສະແດງພາບປະຊາກອນ",
    "Row-level exports": "ສົ່ງອອກຂໍ້ມູນລາຍແຖວ",
    "Methodology & limitations": "ວິທີການ ແລະ ຂໍ້ຈຳກັດ",
    # Forecast overview
    "Time-based validation: 2016–2022 train → 2023–2025 test": (
        "ການກວດສອບຕາມເວລາ: ຝຶກສອນ 2016–2022 → ທົດສອບ 2023–2025"
    ),
    "This chart measures performance on later historical years that each model did "
    "not see during fitting. Model selection uses holdout MAPE, not training fit.": (
        "ແຜນພູມນີ້ວັດຜົນໃນປີຫຼັງໆ ທີ່ແບບຈຳລອງບໍ່ໄດ້ເຫັນໃນຕອນຝຶກສອນ. "
        "ການເລືອກແບບຈຳລອງໃຊ້ MAPE ຂອງຊຸດທົດສອບ, ບໍ່ແມ່ນຄວາມພໍດີຂອງຊຸດຝຶກສອນ."
    ),
    "Accuracy is displayed as 100 − MAPE for continuity with the original project; "
    "it is not classification accuracy.": (
        "ຄວາມຊັດເຈນສະແດງເປັນ 100 − MAPE ເພື່ອໃຫ້ສອດຄ່ອງກັບໂຄງການເດີມ; "
        "ບໍ່ແມ່ນຄວາມຊັດເຈນແບບການຈັດກຸ່ມ (classification accuracy)."
    ),
    "Final fit and forecast: retrained on all 2016–2025 observations": (
        "ຜົນສຸດທ້າຍ ແລະ ການຄາດຄະເນ: ຝຶກສອນຄືນໃໝ່ດ້ວຍຂໍ້ມູນທັງໝົດ 2016–2025"
    ),
    "Actual versus predicted population (2016–2025) by model": (
        "ສົມທຽບລະຫວ່າງຂໍ້ມູນຈິງ ແລະ ຜົນການຄາດຄະເນ (2016–2025), ຕາມແຕ່ລະແບບຈຳລອງ"
    ),
    "Total population forecast 2026–{end} by model": (
        "ສະຫຼຸບການຄາດຄະເນ ປະຊາກອນໂດຍລວມ ປີ 2026–{end}, ຕາມແຕ່ລະແບບຈຳລອງ"
    ),
    "Year": "ປີ",
    "Growth rates and averages": "ອັດຕາການຂະຫຍາຍຕົວ ແລະ ອັດຕາສະເລ່ຍ",
    "{label} average": "ສະເລ່ຍ {label}",
    "Actual": "ຄ່າຈິງ",
    "Each value is the year-over-year growth rate. Actual values run to 2025; model "
    "forecasts continue to the selected end year.": (
        "ແຕ່ລະຄ່າແມ່ນອັດຕາການປ່ຽນແປງ (ເຕີບໂຕ) ທຽບກັບປີຕໍ່ປີ (Year-over-Year). "
        "ຕົວເລກຕົວຈິງແມ່ນຮອດປີ 2025; ຕົວເລກຄາດຄະເນຂອງແບບຈຳລອງແມ່ນສືບຕໍ່ໄປຈົນຮອດປີທີ່ເລືອກໄວ້."
    ),
    # Charts
    "Accuracy of the population forecasts": "ສົມທຽບຄວາມຊັດເຈນຂອງ ຜົນການຄາດຄະເນປະຊາກອນ",
    "Forecast residuals (actual − predicted)": (
        "ຄ່າຜິດດ່ຽງຂອງການຄາດຄະເນ (ຕົວຈິງ − ຜົນການຄາດຄະເນ)"
    ),
    "Actual population": "ປະຊາກອນຕົວຈິງ",
    "Population": "ປະຊາກອນ",
    "Population (people)": "ປະຊາກອນ (ຄົນ)",
    "Residual (actual − predicted)": "ຄ່າຄົງເຫຼືອ (ຕົວຈິງ − ຄາດຄະເນ)",
    "{model} residual": "ຄ່າຄົງເຫຼືອຂອງ {model}",
    "Predicted": "ຄ່າຄາດຄະເນ",
    "{model} prediction": "ຜົນຄາດຄະເນ {model}",
    "Difference": "ຜົນຕ່າງ",
    "Actual vs predicted population (historical fit)": (
        "ປະຊາກອນຈິງ ທຽບກັບ ຄ່າຄາດຄະເນ (ຊ່ວງຂໍ້ມູນປະຫວັດ)"
    ),
    "Prediction": "ຄ່າຄາດຄະເນ",
    "Error": "ຄວາມຜິດພາດ",
    "Holdout year": "ປີຂອງຊຸດທົດສອບ",
    # Pyramid
    "Population pyramid by district": "ປິຣາມິດປະຊາກອນ ຕາມເມືອງ",
    "Compare male and female populations in the age groups 0–4, 5–18, 19–25, 26–40, "
    "41–60, and 61–80+.": (
        "ປຽບທຽບປະຊາກອນຊາຍ ແລະ ຍິງ ໃນກຸ່ມອາຍຸ 0–4, 5–18, 19–25, 26–40, 41–60 ແລະ 61–80+."
    ),
    "The uploaded data must contain an ouName district column.": (
        "ຂໍ້ມູນທີ່ອັບໂຫຼດຕ້ອງມີຖັນ ouName (ຊື່ເມືອງ)."
    ),
    "District": "ເມືອງ",
    "Choose All districts for the nationwide pyramid, or select one district.": (
        "ເລືອກ ທຸກເມືອງ ເພື່ອເບິ່ງປິຣາມິດທົ່ວປະເທດ, ຫຼື ເລືອກເມືອງດຽວ."
    ),
    "Age display": "ການສະແດງອາຍຸ",
    "By age group": "ຕາມກຸ່ມອາຍຸ",
    "By single age": "ຕາມອາຍຸແຕ່ລະປີ",
    "{year} forecast, {model}": "ຄາດຄະເນປີ {year}, {model}",
    "All models is selected, so this future pyramid uses the current lowest-MAPE "
    "model: {model}.": (
        "ເລືອກທຸກແບບຈຳລອງໄວ້, ປິຣາມິດປີອະນາຄົດນີ້ຈຶ່ງໃຊ້ແບບຈຳລອງທີ່ມີ MAPE ຕ່ຳສຸດ: {model}."
    ),
    "The {year} pyramid uses the selected {model} forecast.": (
        "ປິຣາມິດປີ {year} ໃຊ້ຜົນຄາດຄະເນຂອງ {model} ທີ່ເລືອກ."
    ),
    "Male population": "ປະຊາກອນຊາຍ",
    "Female population": "ປະຊາກອນຍິງ",
    "Total population": "ປະຊາກອນທັງໝົດ",
    "Data source: **{source}**.": "ແຫຼ່ງຂໍ້ມູນ: **{source}**.",
    "The totals are the model's forecast, the same as the overview. The bars show each "
    "district–age–sex series rounded to whole people, with series forecast below zero "
    "drawn as 0, so the bars can add up to slightly more than the total.": (
        "ຍອດລວມແມ່ນຜົນຄາດຄະເນຂອງແບບຈຳລອງ ເຊິ່ງກົງກັບພາບລວມ. ແທ່ງສະແດງແຕ່ລະຊຸດຂໍ້ມູນ "
        "ເມືອງ–ອາຍຸ–ເພດ ທີ່ປັດເປັນຈຳນວນຄົນເຕັມ, ຊຸດທີ່ຄາດຄະເນຕ່ຳກວ່າສູນຈະສະແດງເປັນ 0, "
        "ສະນັ້ນຜົນລວມຂອງແທ່ງອາດສູງກວ່າຍອດລວມເລັກນ້ອຍ."
    ),
    "Male": "ຊາຍ",
    "Female": "ຍິງ",
    "Age group": "ກຸ່ມອາຍຸ",
    "Age": "ອາຍຸ",
    "Population pyramid — {location} ({year})": "ປິຣາມິດປະຊາກອນ — {location} ({year})",
    # Map
    "Population map": "ແຜນທີ່ປະຊາກອນ",
    "Total population by province (level 2) or district (level 3). Hover over an area "
    "to see its name and population.": (
        "ປະຊາກອນທັງໝົດ ຕາມແຂວງ (ຂັ້ນ 2) ຫຼື ເມືອງ (ຂັ້ນ 3). "
        "ເລື່ອນເມົາສ໌ໄປເທິງພື້ນທີ່ ເພື່ອເບິ່ງຊື່ ແລະ ຈຳນວນປະຊາກອນ."
    ),
    "Map boundary file not found: {path}": "ບໍ່ພົບໄຟລ໌ຂອບເຂດແຜນທີ່: {path}",
    "The map requires ouID and ouName columns in the data.": (
        "ແຜນທີ່ຕ້ອງການຖັນ ouID ແລະ ouName ໃນຂໍ້ມູນ."
    ),
    "Administrative level": "ຂັ້ນການປົກຄອງ",
    "Province (level 2)": "ແຂວງ (ຂັ້ນ 2)",
    "District (level 3)": "ເມືອງ (ຂັ້ນ 3)",
    "Population by province — {year}": "ປະຊາກອນ ຕາມແຂວງ — {year}",
    "Population by district — {year}": "ປະຊາກອນ ຕາມເມືອງ — {year}",
    "Data source: **{source}**. Nationwide total: **{total}**. Colours use a log scale "
    "so smaller areas can still be told apart.": (
        "ແຫຼ່ງຂໍ້ມູນ: **{source}**. ຍອດລວມທົ່ວປະເທດ: **{total}**. "
        "ສີໃຊ້ມາດຕາສ່ວນ log ເພື່ອໃຫ້ຍັງແຍກພື້ນທີ່ນ້ອຍໄດ້."
    ),
    "{count} area(s) in the data have no boundary on the map.": (
        "ມີ {count} ພື້ນທີ່ໃນຂໍ້ມູນ ທີ່ບໍ່ມີຂອບເຂດໃນແຜນທີ່."
    ),
    "Population table by province": "ຕາຕະລາງປະຊາກອນ ຕາມແຂວງ",
    "Population table by district": "ຕາຕະລາງປະຊາກອນ ຕາມເມືອງ",
    "Name (English)": "ຊື່ (ພາສາອັງກິດ)",
    "Total population and population density by province (level 2) or district "
    "(level 3). Hover over an area to see its name, population and density.": (
        "ປະຊາກອນທັງໝົດ ແລະ ຄວາມໜາແໜ້ນຂອງປະຊາກອນ ຕາມແຂວງ (ຂັ້ນ 2) ຫຼື ເມືອງ (ຂັ້ນ 3). "
        "ເລື່ອນເມົາສ໌ໄປເທິງພື້ນທີ່ ເພື່ອເບິ່ງຊື່, ຈຳນວນປະຊາກອນ ແລະ ຄວາມໜາແໜ້ນ."
    ),
    "Population density by province — {year}": "ຄວາມໜາແໜ້ນຂອງປະຊາກອນ ຕາມແຂວງ — {year}",
    "Population density by district — {year}": "ຄວາມໜາແໜ້ນຂອງປະຊາກອນ ຕາມເມືອງ — {year}",
    "People per km²": "ຄົນ ຕໍ່ ກມ²",
    "Area (km²)": "ເນື້ອທີ່ (ກມ²)",
    "Data source: **{source}**. Nationwide total: **{total}** people, **{density}** "
    "people per km². Colours use a log scale so smaller areas can still be told apart.": (
        "ແຫຼ່ງຂໍ້ມູນ: **{source}**. ຍອດລວມທົ່ວປະເທດ: **{total}** ຄົນ, **{density}** ຄົນ ຕໍ່ ກມ². "
        "ສີໃຊ້ມາດຕາສ່ວນ log ເພື່ອໃຫ້ຍັງແຍກພື້ນທີ່ນ້ອຍໄດ້."
    ),
    "Density divides each area's population by its land area, measured from the map "
    "boundaries.": (
        "ຄວາມໜາແໜ້ນ = ປະຊາກອນຂອງແຕ່ລະພື້ນທີ່ ຫານໃຫ້ເນື້ອທີ່ດິນ ທີ່ວັດຈາກຂອບເຂດໃນແຜນທີ່."
    ),
    "Name (Lao)": "ຊື່ (ພາສາລາວ)",
    # Source data and outliers
    "Forecast source data": "ຂໍ້ມູນຕົ້ນສະບັບສຳລັບການຄາດຄະເນ",
    "Forecasts currently use **{source}**. The original source is preserved and "
    "remains available for comparison.": (
        "ການຄາດຄະເນໃນປັດຈຸບັນໃຊ້ **{source}**. ຂໍ້ມູນຕົ້ນສະບັບຍັງຖືກເກັບໄວ້ "
        "ແລະ ສາມາດໃຊ້ປຽບທຽບໄດ້."
    ),
    "Original filtered data": "ຂໍ້ມູນຕົ້ນສະບັບ (ຕາມຕົວກອງ)",
    "Corrected filtered data": "ຂໍ້ມູນທີ່ແກ້ໄຂແລ້ວ (ຕາມຕົວກອງ)",
    "Conservative outlier correction": "ການແກ້ໄຂຄ່າຜິດປົກກະຕິແບບຮັດກຸມ",
    "Automatic correction uses the selected robust-score threshold of {threshold}. A "
    "value is changed only when a cohort-aware historical anomaly is independently "
    "confirmed by a district-share, sex-ratio, or adjacent-age check.": (
        "ການແກ້ໄຂອັດຕະໂນມັດໃຊ້ເກນຄະແນນທົນທານ (robust score) {threshold} ທີ່ເລືອກ. "
        "ຄ່າຈະຖືກປ່ຽນກໍ່ຕໍ່ເມື່ອຄວາມຜິດປົກກະຕິຕາມປະຫວັດ (ຕາມຮຸ່ນອາຍຸ) ໄດ້ຮັບການຢືນຢັນ "
        "ຈາກການກວດສັດສ່ວນເມືອງ, ອັດຕາສ່ວນເພດ ຫຼື ອາຍຸທີ່ຕິດກັນ."
    ),
    "Outlier correction steps": "ຂັ້ນຕອນການແກ້ໄຂຄ່າຜິດປົກກະຕິ",
    "Corrected observations": "ຈຳນວນຂໍ້ມູນທີ່ແກ້ໄຂ",
    "Affected series": "ຊຸດຂໍ້ມູນທີ່ໄດ້ຮັບຜົນ",
    "Share of historical cells": "ສັດສ່ວນຂອງເຊວທີ່ແກ້ໄຂ",
    "Net historical-cell adjustment": "ການປັບສຸດທິ",
    "No observations in the current filters met the conservative correction rule.": (
        "ບໍ່ມີຂໍ້ມູນໃນຕົວກອງປັດຈຸບັນ ທີ່ເຂົ້າເງື່ອນໄຂການແກ້ໄຂແບບຮັດກຸມ."
    ),
    "Corrections": "ຈຳນວນການແກ້ໄຂ",
    "Download correction audit": "ດາວໂຫຼດບັນທຶກການແກ້ໄຂ",
    "Download corrected source data": "ດາວໂຫຼດຂໍ້ມູນທີ່ແກ້ໄຂແລ້ວ",
    "Forecast impact: original versus corrected data": (
        "ຜົນກະທົບຕໍ່ການຄາດຄະເນ: ຂໍ້ມູນຕົ້ນສະບັບ ທຽບກັບ ຂໍ້ມູນທີ່ແກ້ໄຂແລ້ວ"
    ),
    "This comparison uses the current district, age, and sex filters. It shows how the "
    "conservative corrections affect each of the existing three models.": (
        "ການປຽບທຽບນີ້ໃຊ້ຕົວກອງເມືອງ, ອາຍຸ ແລະ ເພດ ປັດຈຸບັນ. ສະແດງໃຫ້ເຫັນວ່າການແກ້ໄຂ"
        "ແບບຮັດກຸມ ມີຜົນຕໍ່ແຕ່ລະແບບຈຳລອງທັງສາມແນວໃດ."
    ),
    "Forecast year": "ປີຄາດຄະເນ",
    "Original forecast": "ຄາດຄະເນ (ຂໍ້ມູນຕົ້ນສະບັບ)",
    "Corrected-data forecast": "ຄາດຄະເນ (ຂໍ້ມູນທີ່ແກ້ໄຂ)",
    "Change (%)": "ການປ່ຽນແປງ (%)",
    "Original value": "ຄ່າຕົ້ນສະບັບ",
    "Corrected value": "ຄ່າທີ່ແກ້ໄຂ",
    "Temporal robust score": "ຄະແນນທົນທານຕາມເວລາ",
    "Supporting check": "ການກວດສອບທີ່ຢືນຢັນ",
    "Supporting robust score": "ຄະແນນທົນທານທີ່ຢືນຢັນ",
    "Correction method": "ວິທີແກ້ໄຂ",
    "Check": "ການກວດສອບ",
    "Reason": "ເຫດຜົນ",
    "Observed": "ຄ່າທີ່ສັງເກດ",
    "Expected": "ຄ່າຄາດໝາຍ",
    "Unit": "ໜ່ວຍ",
    "Robust score": "ຄະແນນທົນທານ",
    "Nationwide outlier review": "ການທົບທວນຄ່າຜິດປົກກະຕິທົ່ວປະເທດ",
    "Checks always use the complete nationwide dataset as their reference. Sidebar "
    "selections and the year selection below only filter the displayed flags. A flag "
    "indicates a record to review; it is not automatically an error.": (
        "ການກວດສອບໃຊ້ຂໍ້ມູນທົ່ວປະເທດທັງໝົດເປັນເກນອ້າງອິງສະເໝີ. ຕົວກອງໃນແຖບຂ້າງ "
        "ແລະ ການເລືອກປີດ້ານລຸ່ມ ພຽງແຕ່ກອງລາຍການທີ່ສະແດງ. ລາຍການທີ່ຖືກໝາຍ "
        "ແມ່ນຂໍ້ມູນທີ່ຄວນທົບທວນ, ບໍ່ແມ່ນຂໍ້ຜິດພາດສະເໝີໄປ."
    ),
    "Active robust-score threshold": "ເກນຄະແນນທົນທານທີ່ໃຊ້",
    "Years to display": "ປີທີ່ສະແດງ",
    "Select at least one year to display outlier results.": (
        "ເລືອກຢ່າງໜ້ອຍໜຶ່ງປີ ເພື່ອສະແດງຜົນຄ່າຜິດປົກກະຕິ."
    ),
    "Flagged observations": "ຈຳນວນທີ່ຖືກໝາຍ",
    "No observations exceed the selected robust-score threshold.": (
        "ບໍ່ມີຂໍ້ມູນໃດເກີນເກນຄະແນນທົນທານທີ່ເລືອກ."
    ),
    "Flags": "ຈຳນວນທີ່ຖືກໝາຍ",
    "Download flags: {title}": "ດາວໂຫຼດລາຍການທີ່ຖືກໝາຍ: {title}",
    "For each age, sex, and year, compares that group's share of its district "
    "population across all districts using median-based robust scores.": (
        "ສຳລັບແຕ່ລະອາຍຸ, ເພດ ແລະ ປີ, ປຽບທຽບສັດສ່ວນຂອງກຸ່ມນັ້ນໃນປະຊາກອນເມືອງ "
        "ກັບທຸກເມືອງ ໂດຍໃຊ້ຄະແນນທົນທານອີງໃສ່ຄ່າກາງ (median)."
    ),
    "Compares each age 1–79 population with the same birth cohort in the previous "
    "year, then evaluates that change against all districts.": (
        "ປຽບທຽບປະຊາກອນອາຍຸ 1–79 ປີ ກັບຮຸ່ນເກີດດຽວກັນໃນປີກ່ອນ, "
        "ແລ້ວປະເມີນການປ່ຽນແປງນັ້ນທຽບກັບທຸກເມືອງ."
    ),
    "Combines unusual male-share checks with sharp local breaks between adjacent "
    "ages. The open-ended 80+ group is excluded from age-neighbor checks.": (
        "ລວມການກວດສັດສ່ວນເພດຊາຍທີ່ຜິດປົກກະຕິ ກັບການຂາດຕອນຢ່າງແຮງລະຫວ່າງອາຍຸທີ່ຕິດກັນ. "
        "ກຸ່ມອາຍຸ 80+ ບໍ່ໄດ້ນຳມາກວດກັບອາຍຸທີ່ຕິດກັນ."
    ),
    # Fixed values inside the outlier and correction tables
    "Nationwide district comparison": "ປຽບທຽບເມືອງທົ່ວປະເທດ",
    "Historical change": "ການປ່ຽນແປງຕາມປະຫວັດ",
    "Demographic consistency": "ຄວາມສອດຄ່ອງດ້ານປະຊາກອນ",
    "Unusual district age-sex share": "ສັດສ່ວນອາຍຸ–ເພດ ຂອງເມືອງຜິດປົກກະຕິ",
    "Unusual cohort change from the previous age and year": (
        "ການປ່ຽນແປງຂອງຮຸ່ນອາຍຸ ຈາກອາຍຸ ແລະ ປີກ່ອນ ຜິດປົກກະຕິ"
    ),
    "Unusual male share": "ສັດສ່ວນເພດຊາຍຜິດປົກກະຕິ",
    "Unusual break between adjacent ages": "ການຂາດຕອນລະຫວ່າງອາຍຸທີ່ຕິດກັນຜິດປົກກະຕິ",
    "% of district population": "% ຂອງປະຊາກອນເມືອງ",
    "% cohort change": "% ການປ່ຽນແປງຂອງຮຸ່ນອາຍຸ",
    "% male": "% ເພດຊາຍ",
    "% above/below adjacent-age baseline": "% ສູງ/ຕ່ຳກວ່າອາຍຸທີ່ຕິດກັນ",
    "Two-sided log-linear interpolation": "ການປະມານຄ່າແບບ log-linear ສອງດ້ານ",
    "Theil-Sen endpoint estimate": "ການປະເມີນຈຸດປາຍ Theil–Sen",
    # Exports
    "Download the population forecasts as CSV or HTML. The files include every "
    "forecast from Linear Regression, Polynomial Regression and Exponential Growth.": (
        "ຜົນການຄາດຄະເນປະຊາກອນ ສາມາດດາວໂຫຼດໄດ້ໃນແບບ CSV ຫຼື HTML. ປະກອບມີຂໍ້ມູນ"
        "ການຄາດຄະເນທັງໝົດ ຕາມແຕ່ລະແບບ Linear Regression, Polynomial Regression "
        "ແລະ Exponential Growth."
    ),
    "Download all-model CSV": "ດາວໂຫຼດ CSV (ທຸກແບບຈຳລອງ)",
    "Download Polynomial HTML": "ດາວໂຫຼດ HTML (Polynomial)",
    # Methodology
    "Course purpose: help every student understand the end-to-end AI workflow and "
    "apply it to a practical task that addresses a real-world problem.": (
        "ຈຸດປະສົງຂອງຫຼັກສູດ: ຊ່ວຍໃຫ້ນັກຮຽນທຸກຄົນເຂົ້າໃຈຂັ້ນຕອນການເຮັດວຽກຂອງ AI "
        "ແບບຄົບວົງຈອນ ແລະ ນຳໄປໃຊ້ກັບວຽກງານຕົວຈິງ ເພື່ອແກ້ໄຂບັນຫາໃນຊີວິດຈິງ."
    ),
    "CRISP-DM project workflow": "ຂະບວນການ CRISP-DM ຂອງໂຄງການ",
    "Application architecture": "ໂຄງສ້າງຂອງແອັບພລິເຄຊັນ",
    "Limitations and responsible use": "ຂໍ້ຈຳກັດ ແລະ ການນຳໃຊ້ຢ່າງຮັບຜິດຊອບ",
    "Method and software references": "ເອກະສານອ້າງອິງດ້ານວິທີການ ແລະ ຊອບແວ",
    # Three-step layout
    "Step 1: Dataset and sources": "ຂັ້ນຕອນ 1: ຂໍ້ມູນ ແລະ ແຫຼ່ງຂໍ້ມູນ",
    "Step 2: Forecasting and visualization": "ຂັ້ນຕອນ 2: ການຄາດຄະເນ ແລະ ການສະແດງພາບ",
    "Step 3: Summary dashboard": "ຂັ້ນຕອນ 3: ສະຫຼຸບ ແລະ ແຜງຄວບຄຸມ",
    "Dataset and data sources": "ຊຸດຂໍ້ມູນ ແລະ ແຫຼ່ງຂໍ້ມູນ",
    "Population by age, sex, district and year from the Ministry of Health DHIS2 system: "
    "what the data covers, where it comes from and how complete it is.": (
        "ຂໍ້ມູນປະຊາກອນຕາມອາຍຸ, ເພດ, ເມືອງ ແລະ ປີ ຈາກລະບົບ DHIS2 ຂອງກະຊວງສາທາລະນະສຸກ: "
        "ຂອບເຂດຂອງຂໍ້ມູນ, ແຫຼ່ງທີ່ມາ ແລະ ຄວາມຄົບຖ້ວນ."
    ),
    "Forecasting details and visualization": "ລາຍລະອຽດການຄາດຄະເນ ແລະ ການສະແດງພາບ",
    "How the models are built and tested, how outliers are handled, the forecast results, "
    "and maps and charts of the population.": (
        "ວິທີສ້າງ ແລະ ທົດສອບແບບຈຳລອງ, ການຈັດການຄ່າຜິດປົກກະຕິ, ຜົນການຄາດຄະເນ "
        "ແລະ ແຜນທີ່ ແລະ ແຜນພູມຂອງປະຊາກອນ."
    ),
    "Summary and interactive dashboard": "ສະຫຼຸບ ແລະ ແຜງຄວບຄຸມແບບໂຕ້ຕອບ",
    "The headline numbers, key findings and downloads in one place. Everything follows the "
    "sidebar settings and the baseline data type.": (
        "ຕົວເລກຫຼັກ, ຜົນທີ່ພົບ ແລະ ໄຟລ໌ດາວໂຫຼດ ຢູ່ໃນບ່ອນດຽວ. ທຸກຢ່າງປ່ຽນຕາມການຕັ້ງຄ່າ"
        "ໃນແຖບຂ້າງ ແລະ ປະເພດຂໍ້ມູນພື້ນຖານ."
    ),
    "Methodology and models": "ວິທີການ ແລະ ແບບຈຳລອງ",
    "Outlier detection and handling": "ການກວດ ແລະ ແກ້ໄຂຄ່າຜິດປົກກະຕິ",
    "Forecast results": "ຜົນການຄາດຄະເນ",
    "Maps and charts": "ແຜນທີ່ ແລະ ແຜນພູມ",
    "Further analysis": "ການວິເຄາະເພີ່ມເຕີມ",
    "Methodology": "ວິທີການ",
    "Outliers": "ຄ່າຜິດປົກກະຕິ",
    "Results": "ຜົນການຄາດຄະເນ",
    "Maps & charts": "ແຜນທີ່ ແລະ ແຜນພູມ",
    "People per km², {year}": "ຄົນ ຕໍ່ ກມ², {year}",
    "Preview the row-level forecasts": "ເບິ່ງຕົວຢ່າງການຄາດຄະເນລາຍແຖວ",
    # Step 1
    "Records": "ຈຳນວນແຖວ",
    "district–age–sex series": "ຊຸດຂໍ້ມູນ ເມືອງ–ອາຍຸ–ເພດ",
    "{count} provinces": "{count} ແຂວງ",
    "{count} sexes, single years 0–80+": "{count} ເພດ, ອາຍຸແຕ່ລະປີ 0–80+",
    "{count} sexes": "{count} ເພດ",
    "{count} values": "{count} ຄ່າ",
    "Lao PDR": "ສປປ ລາວ",
    "Forecast years": "ປີຄາດຄະເນ",
    "Outlier correction": "ການແກ້ໄຂຄ່າຜິດປົກກະຕິ",
    "threshold {value}": "ເກນ {value}",
    "Filters": "ຕົວກອງ",
    "{count} active": "ເປີດໃຊ້ {count}",
    "All data": "ຂໍ້ມູນທັງໝົດ",
    "Without an upload, the app uses {name}.": "ຖ້າບໍ່ອັບໂຫຼດ, ແອັບຈະໃຊ້ {name}.",
    "Does the forecast follow the past trend?": "ການຄາດຄະເນສອດຄ່ອງກັບແນວໂນ້ມໃນອະດີດບໍ?",
    "Actual {start}–{last} and forecast {first}–{end} ({model}), using the baseline data "
    "and sidebar filters. Hatched bars and dashed lines are forecasts.": (
        "ຂໍ້ມູນຈິງ {start}–{last} ແລະ ຄາດຄະເນ {first}–{end} ({model}), ຕາມຂໍ້ມູນພື້ນຖານ "
        "ແລະ ຕົວກອງໃນແຖບຂ້າງ. ແທ່ງທີ່ມີລາຍຂີດ ແລະ ເສັ້ນປະ ແມ່ນຄ່າຄາດຄະເນ."
    ),
    "Population by year and sex, {start}–{end}": "ປະຊາກອນ ຕາມປີ ແລະ ເພດ, {start}–{end}",
    "Population by single age: {start} vs {end}": "ປະຊາກອນ ຕາມອາຍຸແຕ່ລະປີ: {start} ທຽບກັບ {end}",
    "Dataset and sources": "ຂໍ້ມູນ ແລະ ແຫຼ່ງຂໍ້ມູນ",
    "Forecasting and visualization": "ການຄາດຄະເນ ແລະ ການສະແດງພາບ",
    "Summary dashboard": "ສະຫຼຸບ ແລະ ແຜງຄວບຄຸມ",
    "Use responsibly": "ໃຊ້ຢ່າງຮັບຜິດຊອບ",
    "Limitations": "ຂໍ້ຈຳກັດ",
    "References": "ເອກະສານອ້າງອິງ",
    "Number": "ຈຳນວນ",
    "Share (%)": "ສັດສ່ວນ (%)",
    "Total": "ລວມ",
    "Growth vs previous year": "ເຕີບໂຕທຽບກັບປີກ່ອນ",
    "Map background": "ພື້ນຫຼັງແຜນທີ່",
    "OpenStreetMap": "OpenStreetMap",
    "None": "ບໍ່ມີ",
    "OpenStreetMap needs an internet connection. Choose None to show the maps offline.": (
        "OpenStreetMap ຕ້ອງໃຊ້ອິນເຕີເນັດ. ເລືອກ ບໍ່ມີ ເພື່ອສະແດງແຜນທີ່ແບບອອບລາຍ."
    ),
    "Land area (km²)": "ເນື້ອທີ່ດິນ (ກມ²)",
    "Years": "ປີ",
    "{count} annual values": "{count} ຄ່າລາຍປີ",
    "Land area": "ເນື້ອທີ່ດິນ",
    "measured from the map boundaries": "ວັດຈາກຂອບເຂດໃນແຜນທີ່",
    "Lao PDR: area covered by the data": "ສປປ ລາວ: ພື້ນທີ່ທີ່ຂໍ້ມູນຄອບຄຸມ",
    "Selected districts in Lao PDR": "ເມືອງທີ່ເລືອກ ໃນ ສປປ ລາວ",
    "The whole country (level 1) is shown when all districts are selected. Choose "
    "districts in the sidebar to highlight them. Hover for area, population ({year}) "
    "and density.": (
        "ເມື່ອເລືອກທຸກເມືອງ ຈະສະແດງທົ່ວປະເທດ (ຂັ້ນ 1). ເລືອກເມືອງໃນແຖບຂ້າງ ເພື່ອໄຮໄລທ໌. "
        "ເລື່ອນເມົາສ໌ເພື່ອເບິ່ງເນື້ອທີ່, ປະຊາກອນ ({year}) ແລະ ຄວາມໜາແໜ້ນ."
    ),
    "{count} selected district(s): {area} km², {share} of the country.": (
        "ເລືອກ {count} ເມືອງ: {area} ກມ², ເທົ່າກັບ {share} ຂອງປະເທດ."
    ),
    "Data source": "ແຫຼ່ງຂໍ້ມູນ",
    "Loaded file: {name}": "ໄຟລ໌ທີ່ໂຫຼດ: {name}",
    "Data quality checks": "ການກວດຄຸນນະພາບຂໍ້ມູນ",
    "Completeness": "ຄວາມຄົບຖ້ວນ",
    "{count} empty cells": "{count} ເຊວທີ່ວ່າງ",
    "Unique keys": "ລະຫັດບໍ່ຊ້ຳກັນ",
    "{count} duplicate keys": "{count} ລະຫັດທີ່ຊ້ຳ",
    "Valid values": "ຄ່າທີ່ຖືກຕ້ອງ",
    "numeric and not negative": "ເປັນຕົວເລກ ແລະ ບໍ່ຕິດລົບ",
    "Structural coverage": "ຄວາມຄົບຖ້ວນຂອງໂຄງສ້າງ",
    "districts × ages × sexes": "ເມືອງ × ອາຍຸ × ເພດ",
    "Score": "ຄະແນນ",
    "Detail": "ລາຍລະອຽດ",
    "Charts below use the source data as loaded (before outlier correction), with the "
    "sidebar filters applied.": (
        "ແຜນພູມດ້ານລຸ່ມໃຊ້ຂໍ້ມູນຕົ້ນສະບັບ (ກ່ອນແກ້ໄຂຄ່າຜິດປົກກະຕິ) ຕາມຕົວກອງໃນແຖບຂ້າງ."
    ),
    "Population by year and sex": "ປະຊາກອນ ຕາມປີ ແລະ ເພດ",
    "Population by single age, {year}": "ປະຊາກອນ ຕາມອາຍຸແຕ່ລະປີ, {year}",
    "Largest districts, {year}": "ເມືອງທີ່ມີປະຊາກອນຫຼາຍທີ່ສຸດ, {year}",
    "Top 12 districts in the current selection by population.": (
        "12 ເມືອງທີ່ມີປະຊາກອນຫຼາຍທີ່ສຸດ ໃນການເລືອກປັດຈຸບັນ."
    ),
    "Data dictionary": "ຄຳອະທິບາຍຖັນຂໍ້ມູນ",
    "Column": "ຖັນ",
    "Type": "ປະເພດ",
    "Description": "ຄຳອະທິບາຍ",
    "District identifier in DHIS2": "ລະຫັດເມືອງໃນ DHIS2",
    "District code and name (the first two digits are the province)": (
        "ລະຫັດ ແລະ ຊື່ເມືອງ (ສອງຕົວເລກທຳອິດແມ່ນແຂວງ)"
    ),
    "Data element (Population)": "ອົງປະກອບຂໍ້ມູນ (ປະຊາກອນ)",
    "Age: single years 0–79 and 80+": "ອາຍຸ: ແຕ່ລະປີ 0–79 ແລະ 80+",
    "Sex: Male (M) or Female (F)": "ເພດ: ຊາຍ (M) ຫຼື ຍິງ (F)",
    "Population in each year (one column per year)": "ປະຊາກອນໃນແຕ່ລະປີ (ໜຶ່ງຖັນຕໍ່ໜຶ່ງປີ)",
    # Step 2
    "Forecasting models": "ແບບຈຳລອງການຄາດຄະເນ",
    "Each model is fitted to the yearly population totals, and separately to every "
    "district–age–sex series for the row-level forecasts.": (
        "ແຕ່ລະແບບຈຳລອງຖືກປັບໃຫ້ເຂົ້າກັບຍອດລວມປະຊາກອນລາຍປີ ແລະ ແຍກກັນສຳລັບທຸກຊຸດຂໍ້ມູນ "
        "ເມືອງ–ອາຍຸ–ເພດ ສຳລັບການຄາດຄະເນລາຍແຖວ."
    ),
    "A straight line: about the same number of people is added every year.": (
        "ເສັ້ນຊື່: ປະຊາກອນເພີ່ມຂຶ້ນປະມານຈຳນວນດຽວກັນທຸກປີ."
    ),
    "A degree-2 curve on centred years, so growth can speed up or slow down.": (
        "ເສັ້ນໂຄ້ງລະດັບ 2 ເທິງປີທີ່ປັບໃຫ້ຢູ່ກາງ, ການເຕີບໂຕຈຶ່ງສາມາດໄວຂຶ້ນ ຫຼື ຊ້າລົງໄດ້."
    ),
    "A constant growth rate: about the same percentage is added every year.": (
        "ອັດຕາການເຕີບໂຕຄົງທີ່: ເພີ່ມຂຶ້ນປະມານເປີເຊັນດຽວກັນທຸກປີ."
    ),
    "Test-year MAPE: **{mape}**": "MAPE ປີທົດສອບ: **{mape}**",
    "How the models are tested and used": "ວິທີທົດສອບ ແລະ ນຳໃຊ້ແບບຈຳລອງ",
    "Population change by province, {start}–{end}": "ການປ່ຽນແປງປະຊາກອນ ຕາມແຂວງ, {start}–{end}",
    "Forecast with {model}.": "ຄາດຄະເນດ້ວຍ {model}.",
    "The three models compared, {start}–{end}": "ປຽບທຽບສາມແບບຈຳລອງ, {start}–{end}",
    "How far the models disagree grows with the forecast horizon.": (
        "ຄວາມແຕກຕ່າງລະຫວ່າງແບບຈຳລອງ ຍິ່ງຫຼາຍຂຶ້ນ ເມື່ອຄາດຄະເນໄປໄກຂຶ້ນ."
    ),
    "Most densely populated districts, {year}": "ເມືອງທີ່ມີຄວາມໜາແໜ້ນສູງສຸດ, {year}",
    "People per km², forecast with {model}.": "ຄົນ ຕໍ່ ກມ², ຄາດຄະເນດ້ວຍ {model}.",
    # Step 3
    "Interactive summary dashboard": "ແຜງສະຫຼຸບແບບໂຕ້ຕອບ",
    "Key findings": "ຜົນທີ່ພົບທີ່ສຳຄັນ",
    "Population keeps growing": "ປະຊາກອນຍັງເພີ່ມຂຶ້ນຕໍ່ເນື່ອງ",
    "From {start} in {start_year} to {end} in {end_year} ({change}), about {rate} a year.": (
        "ຈາກ {start} ໃນປີ {start_year} ເປັນ {end} ໃນປີ {end_year} ({change}), "
        "ປະມານ {rate} ຕໍ່ປີ."
    ),
    "Most accurate model: {model}": "ແບບຈຳລອງທີ່ຊັດເຈນທີ່ສຸດ: {model}",
    "Test-year MAPE {mape}. Other models: {others}.": (
        "MAPE ປີທົດສອບ {mape}. ແບບຈຳລອງອື່ນ: {others}."
    ),
    "The population is getting older": "ປະຊາກອນມີອາຍຸສູງຂຶ້ນ",
    "People aged 41 and over: {start} of the population in {start_year}, {end} in {end_year}.": (
        "ຜູ້ທີ່ມີອາຍຸ 41 ປີຂຶ້ນໄປ: {start} ຂອງປະຊາກອນໃນປີ {start_year}, {end} ໃນປີ {end_year}."
    ),
    "Density rises": "ຄວາມໜາແໜ້ນເພີ່ມຂຶ້ນ",
    "From {start} to {end} people per km² between {start_year} and {end_year}.": (
        "ຈາກ {start} ເປັນ {end} ຄົນ ຕໍ່ ກມ² ລະຫວ່າງປີ {start_year} ແລະ {end_year}."
    ),
    "Fastest-growing provinces": "ແຂວງທີ່ເຕີບໂຕໄວທີ່ສຸດ",
    "Outliers corrected": "ຄ່າຜິດປົກກະຕິທີ່ແກ້ໄຂ",
    "{count} values ({share}) were corrected; the original values are kept in the audit log.": (
        "ແກ້ໄຂ {count} ຄ່າ ({share}); ຄ່າຕົ້ນສະບັບຍັງເກັບໄວ້ໃນບັນທຶກການກວດສອບ."
    ),
    "Summary statistics": "ສະຖິຕິສະຫຼຸບ",
    "By year": "ຕາມປີ",
    "By province": "ຕາມແຂວງ",
    "Annual growth (%)": "ອັດຕາເຕີບໂຕຕໍ່ປີ (%)",
    "Province": "ແຂວງ",
    "Population {year}": "ປະຊາກອນ {year}",
    "Forecast {year}": "ຄາດຄະເນ {year}",
    "Change": "ການປ່ຽນແປງ",
    "People per km² ({year})": "ຄົນ ຕໍ່ ກມ² ({year})",
    "Downloads and exports": "ດາວໂຫຼດ ແລະ ສົ່ງອອກ",
    "Summary and audit files": "ໄຟລ໌ສະຫຼຸບ ແລະ ບັນທຶກການກວດສອບ",
    "Summary by year (CSV)": "ສະຫຼຸບຕາມປີ (CSV)",
    "Summary by province (CSV)": "ສະຫຼຸບຕາມແຂວງ (CSV)",
    "About this forecast: limitations and references": (
        "ກ່ຽວກັບການຄາດຄະເນນີ້: ຂໍ້ຈຳກັດ ແລະ ເອກະສານອ້າງອິງ"
    ),
    "Areas and densities use land areas measured from the DHIS2 map boundaries, which "
    "total about 2% less than the official 236,800 km².": (
        "ເນື້ອທີ່ ແລະ ຄວາມໜາແໜ້ນ ໃຊ້ເນື້ອທີ່ດິນທີ່ວັດຈາກຂອບເຂດແຜນທີ່ DHIS2 "
        "ເຊິ່ງລວມກັນໜ້ອຍກວ່າ 236,800 ກມ² ທາງການ ປະມານ 2%."
    ),
}

BLOCKS: dict[str, dict[str, str]] = {
    "use_responsibly": {
        "en": """
- The forecasts are based on only 10 values (10 years) of data.
- Migration, births, deaths and policy changes are not used as explicit predictors in this forecast.
- Original values and the correction audits must remain available for review.
- The forecasts can support planning, but they do not replace official projections.
""",
        "lo": """
- ຂໍ້ມູນທີ່ນຳມາເປັນຖານໃນການພະຍາກອນ ມີ 10 ຄ່າ (10 ປີ) ເທົ່ານັ້ນ.
- ປັດໄຈ: ການຍ້າຍຖິ່ນ, ການເກີດ, ການເສຍຊີວິດ ແລະ ການປ່ຽນແປງນະໂຍບາຍ ບໍ່ໄດ້ນຳມາພິຈາລະນາໃນການພະຍາກອນຄັ້ງນີ້ (explicit predictors).
- ຄ່າຕົ້ນສະບັບ ແລະ ການກວດສອບການແກ້ໄຂ (correction audits) ຕ້ອງສາມາດທົບທວນໄດ້.
- ຜົນຄາດຄະເນສາມາດຊ່ວຍສະໜັບສະໜູນການວາງແຜນ, ແຕ່ບໍ່ໄດ້ທົດແທນການຄາດຄະເນທາງການ.
""",
    },
    "data_source": {
        "en": """
- **System:** DHIS2 (District Health Information Software 2), Ministry of Health, Lao PDR.
- **Data element:** Population, by district, single year of age (0–79, 80+) and sex.
- **Coverage:** 148 districts in 18 provinces, every year from 2016 to 2025.
- **Format:** one CSV row per district–age–sex series, one column per year.
- **Map boundaries:** DHIS2 organisation units (country, provinces, districts), linked by `ouID`.
""",
        "lo": """
- **ລະບົບ:** DHIS2 (District Health Information Software 2), ກະຊວງສາທາລະນະສຸກ, ສປປ ລາວ.
- **ອົງປະກອບຂໍ້ມູນ:** ປະຊາກອນ ຕາມເມືອງ, ອາຍຸແຕ່ລະປີ (0–79, 80+) ແລະ ເພດ.
- **ຂອບເຂດ:** 148 ເມືອງ ໃນ 18 ແຂວງ, ທຸກປີແຕ່ 2016 ຫາ 2025.
- **ຮູບແບບ:** ໜຶ່ງແຖວ CSV ຕໍ່ໜຶ່ງຊຸດຂໍ້ມູນ ເມືອງ–ອາຍຸ–ເພດ, ໜຶ່ງຖັນຕໍ່ໜຶ່ງປີ.
- **ຂອບເຂດແຜນທີ່:** ຫົວໜ່ວຍການຈັດຕັ້ງ DHIS2 (ປະເທດ, ແຂວງ, ເມືອງ), ເຊື່ອມດ້ວຍ `ouID`.
""",
    },
    "validation_scheme": {
        "en": """
1. **Train** each model on 2016–2022.
2. **Test** on 2023–2025, which the models never saw, using MAE, RMSE, MAPE and R².
3. **Select** the model with the lowest test-year MAPE.
4. **Retrain** on all of 2016–2025 and forecast from 2026 to the chosen end year.
5. **Row level:** the same models are fitted to each of the 23,976 district–age–sex series;
   for the Linear and Polynomial models these add up exactly to the national forecast.
""",
        "lo": """
1. **ຝຶກສອນ** ແຕ່ລະແບບຈຳລອງດ້ວຍຂໍ້ມູນ 2016–2022.
2. **ທົດສອບ** ດ້ວຍຂໍ້ມູນ 2023–2025 ທີ່ແບບຈຳລອງບໍ່ເຄີຍເຫັນ, ວັດດ້ວຍ MAE, RMSE, MAPE ແລະ R².
3. **ເລືອກ** ແບບຈຳລອງທີ່ມີ MAPE ປີທົດສອບຕ່ຳທີ່ສຸດ.
4. **ຝຶກສອນຄືນໃໝ່** ດ້ວຍຂໍ້ມູນທັງໝົດ 2016–2025 ແລະ ຄາດຄະເນແຕ່ປີ 2026 ຫາ ປີທີ່ເລືອກ.
5. **ລາຍແຖວ:** ແບບຈຳລອງດຽວກັນຖືກປັບໃຫ້ແຕ່ລະຊຸດຂໍ້ມູນ ເມືອງ–ອາຍຸ–ເພດ ທັງ 23,976 ຊຸດ;
   ສຳລັບ Linear ແລະ Polynomial ຜົນລວມເທົ່າກັບການຄາດຄະເນທົ່ວປະເທດພໍດີ.
""",
    },
    "demo_path": {
        "en": """
1. **Forecast Overview**: show holdout validation, the selected model and the national forecast.
2. **Source data & outliers**: explain the nationwide correction summary and audit trail.
3. **Population Visualization**: switch the pyramid and map between the whole country and one area.
4. **Row-level exports**: show the reusable CSV and HTML outputs.

The methodology tab is supporting material; it does not need to be opened during the timed demo.
""",
        "lo": """
1. **ພາບລວມການຄາດຄະເນ**: ສະແດງການກວດສອບດ້ວຍຊຸດທົດສອບ, ແບບຈຳລອງທີ່ເລືອກ ແລະ ຜົນຄາດຄະເນທົ່ວປະເທດ.
2. **ຂໍ້ມູນຕົ້ນສະບັບ ແລະ ຄ່າຜິດປົກກະຕິ**: ອະທິບາຍສະຫຼຸບການແກ້ໄຂທົ່ວປະເທດ ແລະ ບັນທຶກການກວດສອບ.
3. **ການສະແດງພາບປະຊາກອນ**: ສະຫຼັບປິຣາມິດ ແລະ ແຜນທີ່ ລະຫວ່າງທົ່ວປະເທດ ແລະ ພື້ນທີ່ດຽວ.
4. **ສົ່ງອອກຂໍ້ມູນລາຍແຖວ**: ສະແດງໄຟລ໌ CSV ແລະ HTML ທີ່ນຳໃຊ້ຄືນໄດ້.

ແຖບວິທີການ ແລະ ຂໍ້ຈຳກັດ ແມ່ນເອກະສານປະກອບ; ບໍ່ຈຳເປັນຕ້ອງເປີດໃນລະຫວ່າງການສາທິດ.
""",
    },
    "correction_steps": {
        "en": """
1. Run the nationwide median/MAD outlier checks on the complete, unfiltered dataset.
2. Require two independent signals: a historical cohort-change flag and at least one
   demographic consistency or nationwide district-share flag.
3. For an internal year, replace the confirmed value with **two-sided log-linear
   interpolation** from the unchanged previous and following years:
   `exp((log(1 + previous) + log(1 + following)) / 2) - 1`.
4. For 2016 or 2025, use a **Theil–Sen endpoint estimate** calculated from the remaining
   years' median pairwise log-growth slopes.
5. Correct no more than two non-adjacent years in one district–age–sex series, clamp at
   zero, and round to a whole person. The original values are retained in the audit log.
""",
        "lo": """
1. ກວດຫາຄ່າຜິດປົກກະຕິດ້ວຍຄ່າກາງ/MAD (median/MAD) ທົ່ວປະເທດ ໃນຊຸດຂໍ້ມູນຄົບຖ້ວນທີ່ບໍ່ຖືກກອງ.
2. ຕ້ອງມີ 2 ສັນຍານທີ່ເປັນອິດສະຫຼະ: ການໝາຍການປ່ຽນແປງຂອງຮຸ່ນອາຍຸຕາມປະຫວັດ ແລະ ຢ່າງໜ້ອຍ
   ໜຶ່ງການໝາຍດ້ານຄວາມສອດຄ່ອງຂອງປະຊາກອນ ຫຼື ສັດສ່ວນເມືອງທົ່ວປະເທດ.
3. ສຳລັບປີພາຍໃນ, ແທນຄ່າທີ່ຢືນຢັນແລ້ວດ້ວຍ **ການປະມານຄ່າແບບ log-linear ສອງດ້ານ**
   ຈາກປີກ່ອນ ແລະ ປີຖັດໄປທີ່ບໍ່ຖືກປ່ຽນ:
   `exp((log(1 + previous) + log(1 + following)) / 2) - 1`.
4. ສຳລັບປີ 2016 ຫຼື 2025, ໃຊ້ **ການປະເມີນຈຸດປາຍ Theil–Sen** ທີ່ຄຳນວນຈາກຄ່າກາງ
   ຂອງຄວາມຊັນການເຕີບໂຕແບບ log ລາຍຄູ່ ຂອງປີທີ່ເຫຼືອ.
5. ແກ້ໄຂບໍ່ເກີນ 2 ປີທີ່ບໍ່ຕິດກັນ ໃນໜຶ່ງຊຸດຂໍ້ມູນ ເມືອງ–ອາຍຸ–ເພດ, ຈຳກັດຄ່າຕ່ຳສຸດເປັນສູນ
   ແລະ ປັດເປັນຈຳນວນຄົນເຕັມ. ຄ່າຕົ້ນສະບັບຍັງຖືກເກັບໄວ້ໃນບັນທຶກການກວດສອບ.
""",
    },
    "crisp_dm": {
        "en": """
1. **Business understanding**: provide clearer population targets for health-service planning.
2. **Data understanding**: review nationwide DHIS2 population data by district, age, sex, and year.
3. **Data preparation**: validate the schema and treat only independently confirmed outliers.
4. **Modeling**: compare Linear, degree-2 Polynomial, and Exponential Growth models.
5. **Evaluation**: train on 2016–2022 and test on unseen 2023–2025 observations.
6. **Deployment**: retrain on the full 2016–2025 history and deliver forecasts through Streamlit and downloadable exports.
""",
        "lo": """
1. **ເຂົ້າໃຈບັນຫາ**: ໃຫ້ຕົວເລກເປົ້າໝາຍປະຊາກອນທີ່ຊັດເຈນຂຶ້ນ ສຳລັບການວາງແຜນບໍລິການສາທາລະນະສຸກ.
2. **ເຂົ້າໃຈຂໍ້ມູນ**: ທົບທວນຂໍ້ມູນປະຊາກອນ DHIS2 ທົ່ວປະເທດ ຕາມເມືອງ, ອາຍຸ, ເພດ ແລະ ປີ.
3. **ກະກຽມຂໍ້ມູນ**: ກວດສອບໂຄງສ້າງຂໍ້ມູນ ແລະ ແກ້ໄຂສະເພາະຄ່າຜິດປົກກະຕິທີ່ຢືນຢັນແລ້ວ.
4. **ສ້າງແບບຈຳລອງ**: ປຽບທຽບແບບຈຳລອງ Linear, Polynomial ລະດັບ 2 ແລະ Exponential Growth.
5. **ປະເມີນຜົນ**: ຝຶກສອນດ້ວຍຂໍ້ມູນ 2016–2022 ແລະ ທົດສອບດ້ວຍຂໍ້ມູນ 2023–2025 ທີ່ບໍ່ເຄີຍເຫັນ.
6. **ນຳໃຊ້**: ຝຶກສອນຄືນໃໝ່ດ້ວຍຂໍ້ມູນທັງໝົດ 2016–2025 ແລະ ສົ່ງຜົນຄາດຄະເນຜ່ານ Streamlit ແລະ ໄຟລ໌ດາວໂຫຼດ.
""",
    },
    "architecture": {
        "en": (
            "DHIS2 / CSV\n"
            "      ↓\n"
            "Schema validation and nationwide outlier review\n"
            "      ↓\n"
            "Original or conservatively corrected baseline\n"
            "      ↓\n"
            "Time-based validation and model comparison\n"
            "      ↓\n"
            "Full-history retraining and 2026+ forecast\n"
            "      ↓\n"
            "Streamlit charts, audit tables, CSV and HTML exports"
        ),
        "lo": (
            "DHIS2 / CSV\n"
            "      ↓\n"
            "ກວດສອບໂຄງສ້າງຂໍ້ມູນ ແລະ ທົບທວນຄ່າຜິດປົກກະຕິທົ່ວປະເທດ\n"
            "      ↓\n"
            "ຂໍ້ມູນພື້ນຖານ: ຕົ້ນສະບັບ ຫຼື ແກ້ໄຂແບບຮັດກຸມ\n"
            "      ↓\n"
            "ກວດສອບຕາມເວລາ ແລະ ປຽບທຽບແບບຈຳລອງ\n"
            "      ↓\n"
            "ຝຶກສອນຄືນໃໝ່ດ້ວຍຂໍ້ມູນທັງໝົດ ແລະ ຄາດຄະເນປີ 2026 ເປັນຕົ້ນໄປ\n"
            "      ↓\n"
            "ແຜນພູມ Streamlit, ຕາຕະລາງກວດສອບ, ສົ່ງອອກ CSV ແລະ HTML"
        ),
    },
    "limitations": {
        "en": """
- Only ten annual observations are available, so long-horizon forecasts are uncertain.
- The models extrapolate historical trends and do not directly model migration, policy changes, disasters, epidemics, fertility, or mortality.
- Polynomial extrapolation can become unrealistic as the forecast horizon grows.
- Automated outlier treatment can hide a genuine demographic event; original values and the correction audit are therefore retained.
- Prediction intervals are not yet included.
- Results are decision-support estimates and should not replace official projections or demographic expert review.
""",
        "lo": """
- ມີຂໍ້ມູນພຽງ 10 ປີເທົ່ານັ້ນ, ສະນັ້ນການຄາດຄະເນໄລຍະຍາວຍັງບໍ່ແນ່ນອນ.
- ແບບຈຳລອງຕໍ່ແນວໂນ້ມຈາກອະດີດ ແລະ ບໍ່ໄດ້ຄິດໄລ່ການຍ້າຍຖິ່ນ, ການປ່ຽນແປງນະໂຍບາຍ, ໄພພິບັດ, ການລະບາດ, ການເກີດ ຫຼື ການເສຍຊີວິດໂດຍກົງ.
- ການຕໍ່ແນວໂນ້ມແບບ Polynomial ອາດບໍ່ສົມເຫດສົມຜົນ ເມື່ອຄາດຄະເນໄປໄກຂຶ້ນ.
- ການແກ້ໄຂຄ່າຜິດປົກກະຕິອັດຕະໂນມັດ ອາດປິດບັງເຫດການດ້ານປະຊາກອນທີ່ເກີດຂຶ້ນແທ້; ສະນັ້ນຈຶ່ງເກັບຄ່າຕົ້ນສະບັບ ແລະ ບັນທຶກການແກ້ໄຂໄວ້.
- ຍັງບໍ່ມີຊ່ວງການຄາດຄະເນ (prediction intervals).
- ຜົນທີ່ໄດ້ແມ່ນຕົວເລກປະມານເພື່ອຊ່ວຍການຕັດສິນໃຈ ແລະ ບໍ່ຄວນໃຊ້ແທນການຄາດຄະເນທາງການ ຫຼື ການທົບທວນຂອງຜູ້ຊ່ຽວຊານດ້ານປະຊາກອນ.
""",
    },
    "references": {
        "en": """
- Chapman et al. (2000), *CRISP-DM 1.0: Step-by-step data mining guide*.
- Theil (1950) and Sen (1968), robust median-slope estimation.
- scikit-learn documentation: `LinearRegression`, regression metrics, and model evaluation.
- Plotly and Streamlit documentation for interactive visualization and application delivery.
- Population source: DHIS2, Ministry of Health dataset used by this application.
""",
        "lo": """
- Chapman ແລະ ຄະນະ (2000), *CRISP-DM 1.0: Step-by-step data mining guide*.
- Theil (1950) ແລະ Sen (1968), ການປະເມີນຄວາມຊັນດ້ວຍຄ່າກາງແບບທົນທານ.
- ເອກະສານ scikit-learn: `LinearRegression`, ຕົວຊີ້ວັດການຖົດຖອຍ ແລະ ການປະເມີນແບບຈຳລອງ.
- ເອກະສານ Plotly ແລະ Streamlit ສຳລັບແຜນພູມແບບໂຕ້ຕອບ ແລະ ການສ້າງແອັບພລິເຄຊັນ.
- ແຫຼ່ງຂໍ້ມູນປະຊາກອນ: ຊຸດຂໍ້ມູນ DHIS2 ຂອງກະຊວງສາທາລະນະສຸກ ທີ່ແອັບນີ້ນຳໃຊ້.
""",
    },
}


def translate(text: str, language: str, **values: object) -> str:
    """Return ``text`` in ``language`` ("lo" or "en"), formatted with ``values``."""
    result = LAO.get(text, text) if language == "lo" else text
    return result.format(**values) if values else result


def block(name: str, language: str) -> str:
    """Return a long markdown passage in ``language``."""
    passage = BLOCKS[name]
    return passage.get(language, passage["en"])
