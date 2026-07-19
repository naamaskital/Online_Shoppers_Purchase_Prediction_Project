# Online Shoppers Purchase Prediction Project

פרויקט בלמידת מכונה לחיזוי כוונת רכישה של מבקרים באתר קניות אונליין.

## מטרת הפרויקט

מטרת הפרויקט היא לחזות האם ביקור של משתמש באתר קניות אונליין יסתיים ברכישה או לא.

עמודת המטרה היא `Revenue`, ולכן מדובר בבעיית סיווג בינארי.

## תיאור הדאטה

הפרויקט מבוסס על Online Shoppers Purchasing Intention Dataset.

הדאטה כולל 12,330 ביקורים באתר קניות אונליין.

כל שורה מייצגת ביקור אחד באתר, והמאפיינים מתארים את התנהגות המשתמש במהלך הביקור.

המאפיינים כוללים בין היתר:

- מספר דפים שבהם המשתמש ביקר
- משך השהייה בדפים
- שיעורי יציאה ונטישה
- ערך הדפים
- חודש הביקור
- סוג המשתמש
- מקור התנועה
- האם הביקור היה בסוף שבוע

## שאלות המחקר

1. האם ניתן לחזות מראש האם מבקר באתר יבצע רכישה לפי ההתנהגות שלו באתר?
2. אילו מאפיינים משפיעים יותר על הסיכוי לרכישה?
3. איזה מודל למידת מכונה נותן את הביצועים הטובים ביותר עבור הבעיה?
4. כיצד חוסר האיזון בין רכישות לאי רכישות משפיע על ביצועי המודלים?

## המודלים שנבדקו

בפרויקט נבדקו המודלים הבאים:

- Logistic Regression
- KNN
- Decision Tree
- Random Forest
- SVM
- AdaBoost

בנוסף, בוצעה בדיקה של השפעת חוסר איזון בין המחלקות, ונעשה ניסיון לשפר את Random Forest באמצעות Hyperparameter Tuning ו-Threshold Tuning.

## מדדי הערכה

מכיוון שהדאטה אינו מאוזן, לא הסתפקתי ב-Accuracy בלבד.

נבדקו המדדים הבאים:

- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC

## תוצאה מרכזית

המודל הטוב ביותר היה Random Forest.

התוצאה המרכזית של Random Forest:

- Accuracy: 0.8901
- Precision: 0.6312
- Recall: 0.6990
- F1-score: 0.6634
- ROC AUC: 0.9191

Random Forest נתן את האיזון הטוב ביותר בין זיהוי רכישות אמיתיות לבין הימנעות מתחזיות שגויות.

## מבנה הפרויקט

```text
Online_Shoppers_Purchase_Prediction_Project/

├── data/
│   └── online_shoppers_intention.csv

├── results/
│   ├── model_comparison.csv
│   ├── imbalance_comparison.csv
│   ├── random_forest_feature_importance.csv
│   └── graphs and confusion matrices

├── report/
│   ├── final_report_hebrew.md
│   ├── report_summary_hebrew.md
│   └── model_improvement_section_hebrew.md

├── 02_eda.py
├── 03_train_models.py
├── 04_feature_importance.py
├── 05_imbalance_comparison.py
├── 08_improve_random_forest.py
├── 09_course_extensions.py
├── final_project_code.py
├── main.py
├── requirements.txt
└── README.md
```


## מסמכי הגשה

בתיקיית `report/` קיימים כמה מסמכים:

- `final_report_hebrew.md` — הדוח הסופי של הפרויקט.
- `project_full_explanation_hebrew.md` — דוח הסבר מלא, מפורט ואינטואיטיבי מהתחלה עד הסוף.
- `report_summary_hebrew.md` — סיכום קצר של הפרויקט.
- `model_improvement_section_hebrew.md` — פירוט ניסיון שיפור המודל.

## איך להריץ את הפרויקט

להתקנת הספריות הדרושות:

```bash
pip install -r requirements.txt
```

להרצת ניתוח ראשוני של המאגר:

```bash
python3 main.py
```

להרצת שלבי הניתוח והאימון המלאים לפי הסדר:

```bash
python3 02_eda.py
python3 03_train_models.py
python3 04_feature_importance.py
python3 05_imbalance_comparison.py
python3 08_improve_random_forest.py
python3 09_course_extensions.py
```



## הרחבות נוספות

בנוסף למודלים המרכזיים, נוספו שתי הרחבות שמחוברות לחומר הקורס:

- AdaBoost — אלגוריתם הגברה.
- PCA — הורדת ממד לצורך אנליזה ויזואלית של הדאטה.

AdaBoost נתן ביצועים טובים, אך Random Forest נשאר המודל הטוב ביותר לפי F1-score.  
PCA הראה שאין הפרדה דו-ממדית מושלמת בין רכישות לאי רכישות, ולכן הבעיה דורשת שילוב של מספר מאפיינים.

## מסקנה

ניתן לחזות כוונת רכישה ברמה טובה יחסית לפי נתוני התנהגות של המשתמש באתר.

המאפיינים החשובים ביותר קשורים לערך הדפים, משך השהייה בדפי מוצר, מספר דפי המוצר ושיעורי היציאה.
