import os
import pandas as pd

os.makedirs("report", exist_ok=True)

model_comparison = pd.read_csv("results/model_comparison.csv")
imbalance_comparison = pd.read_csv("results/imbalance_comparison.csv")
feature_importance = pd.read_csv("results/random_forest_feature_importance.csv")

best_model = model_comparison.sort_values(by="F1", ascending=False).iloc[0]
top_features = feature_importance.head(10)

with open("report/final_report_hebrew.md", "w", encoding="utf-8") as f:
    f.write("# פרויקט בלמידת מכונה: חיזוי כוונת רכישה באתר קניות אונליין\n\n")

    f.write("## 1. מבוא\n")
    f.write(
        "בפרויקט זה נבחנה היכולת לחזות האם ביקור של משתמש באתר קניות אונליין "
        "יסתיים ברכישה או לא. הבעיה היא בעיית סיווג בינארי, משום שעמודת המטרה "
        "Revenue מקבלת שני ערכים אפשריים: רכישה או אי רכישה.\n\n"
    )

    f.write(
        "הדאטה כולל 12,330 ביקורים באתר קניות אונליין. כל שורה מייצגת ביקור אחד באתר, "
        "והמאפיינים מתארים את התנהגות המשתמש במהלך הביקור, למשל מספר הדפים שבהם ביקר, "
        "משך השהייה בדפים, שיעורי יציאה, סוג המשתמש, מקור התנועה, חודש הביקור ועוד.\n\n"
    )

    f.write("## 2. מטרות ושאלות המחקר\n")
    f.write("בפרויקט נבדקו השאלות הבאות:\n\n")
    f.write("1. האם ניתן לחזות מראש האם מבקר באתר יבצע רכישה לפי ההתנהגות שלו באתר?\n")
    f.write("2. אילו מאפיינים משפיעים יותר על הסיכוי לרכישה?\n")
    f.write("3. איזה מודל למידת מכונה נותן את הביצועים הטובים ביותר עבור הבעיה?\n")
    f.write("4. כיצד חוסר האיזון בין רכישות לאי רכישות משפיע על ביצועי המודלים?\n\n")

    f.write("## 3. תיאור הדאטה\n")
    f.write(
        "הדאטה כולל 18 עמודות: 17 עמודות קלט ועמודת מטרה אחת בשם Revenue. "
        "עמודת Revenue מציינת האם הביקור הסתיים ברכישה.\n\n"
    )

    f.write("התפלגות עמודת המטרה:\n\n")
    f.write("- 10,422 ביקורים לא הסתיימו ברכישה.\n")
    f.write("- 1,908 ביקורים הסתיימו ברכישה.\n")
    f.write("- כלומר, רק כ-15.47% מהביקורים הסתיימו ברכישה.\n\n")

    f.write(
        "מכאן ניתן לראות שהדאטה אינו מאוזן. רוב הדוגמאות שייכות למחלקת אי הרכישה, "
        "ורק מיעוט מהדוגמאות שייכות למחלקת הרכישה.\n\n"
    )

    f.write("## 4. עיבוד מקדים\n")
    f.write(
        "לפני אימון המודלים בוצע עיבוד מקדים לנתונים. העמודות המספריות עברו נרמול "
        "באמצעות StandardScaler, והעמודות הקטגוריאליות עברו קידוד באמצעות OneHotEncoder. "
        "בנוסף, הדאטה חולק לסט אימון וסט בדיקה ביחס של 80% לאימון ו-20% לבדיקה.\n\n"
    )

    f.write(
        "בחלוקה ל-Train ו-Test נעשה שימוש ב-stratify כדי לשמור על יחס דומה בין רכישות "
        "לאי רכישות גם בסט האימון וגם בסט הבדיקה. פעולה זו חשובה במיוחד כאשר הדאטה אינו מאוזן.\n\n"
    )

    f.write("## 5. המודלים שנבדקו\n")
    f.write("בפרויקט נבדקו חמישה מודלים של למידת מכונה:\n\n")
    f.write("- Logistic Regression\n")
    f.write("- KNN\n")
    f.write("- Decision Tree\n")
    f.write("- Random Forest\n")
    f.write("- SVM\n\n")

    f.write(
        "המודלים נבחרו כדי להשוות בין מודלים פשוטים יחסית לבין מודלים מורכבים יותר. "
        "Logistic Regression שימש כמודל בסיסי, בעוד Random Forest מאפשר ללמוד קשרים מורכבים יותר בין המאפיינים.\n\n"
    )

    f.write("## 6. מדדי הערכה\n")
    f.write(
        "מכיוון שהדאטה אינו מאוזן, Accuracy לבדו אינו מספיק להערכת איכות המודל. "
        "מודל יכול לקבל Accuracy גבוה גם אם הוא כמעט תמיד מנבא שאין רכישה. "
        "לכן נבדקו גם המדדים Precision, Recall, F1-score ו-ROC AUC.\n\n"
    )

    f.write("פירוש המדדים:\n\n")
    f.write("- Accuracy: אחוז התחזיות הנכונות מתוך כלל הדוגמאות.\n")
    f.write("- Precision: מתוך כל הדוגמאות שהמודל חזה כרכישה, כמה באמת היו רכישה.\n")
    f.write("- Recall: מתוך כל הרכישות האמיתיות, כמה המודל הצליח לזהות.\n")
    f.write("- F1-score: מדד משולב בין Precision ו-Recall.\n")
    f.write("- ROC AUC: מדד כללי ליכולת ההפרדה של המודל בין שתי המחלקות.\n\n")

    f.write("## 7. תוצאות השוואת המודלים\n")
    f.write("טבלת התוצאות שהתקבלה:\n\n")
    f.write(model_comparison.to_markdown(index=False))
    f.write("\n\n")

    f.write("מתוך הטבלה ניתן לראות כי המודל שקיבל את ערך ה-F1 הגבוה ביותר הוא ")
    f.write(f"{best_model['Model']}. ")
    f.write(
        f"מודל זה הגיע ל-Accuracy של {best_model['Accuracy']:.4f}, "
        f"Precision של {best_model['Precision']:.4f}, "
        f"Recall של {best_model['Recall']:.4f}, "
        f"F1 של {best_model['F1']:.4f}, "
        f"ROC AUC של {best_model['ROC_AUC']:.4f}.\n\n"
    )

    f.write(
        "לכן, Random Forest נבחר כמודל המוביל בפרויקט. הוא נתן איזון טוב יחסית בין "
        "יכולת לזהות רכישות לבין שמירה על דיוק בתחזיות הרכישה.\n\n"
    )

    f.write("## 8. השפעת חוסר האיזון בדאטה\n")
    f.write("כדי לבדוק את השפעת חוסר האיזון, נבדקו גם מודלים ללא איזון וגם מודלים עם class_weight='balanced'.\n\n")
    f.write("טבלת השוואת השפעת האיזון:\n\n")
    f.write(imbalance_comparison.to_markdown(index=False))
    f.write("\n\n")

    f.write(
        "ניתן לראות שמודל Dummy, אשר מנבא תמיד את המחלקה הנפוצה ביותר, הגיע ל-Accuracy של כ-0.845. "
        "למרות זאת, ה-Precision, Recall וה-F1 שלו עבור מחלקת הרכישה היו 0. "
        "כלומר, הוא לא הצליח לזהות אף רכישה בפועל.\n\n"
    )

    f.write(
        "ממצא זה מראה ש-Accuracy עלול להטעות כאשר הדאטה אינו מאוזן. "
        "לכן בפרויקט יש חשיבות רבה למדדים כמו Recall ו-F1, ולא רק ל-Accuracy.\n\n"
    )

    f.write(
        "בנוסף, ניתן לראות שבחלק מהמודלים שימוש ב-class_weight='balanced' שיפר משמעותית את ה-Recall. "
        "לדוגמה, ב-Logistic Regression ה-Recall עלה מ-0.3586 ל-0.7435. "
        "כלומר, המודל המאוזן הצליח לזהות הרבה יותר ביקורים שהסתיימו ברכישה.\n\n"
    )

    f.write("## 9. חשיבות מאפיינים\n")
    f.write("כדי להבין אילו מאפיינים משפיעים יותר על הסיכוי לרכישה, נבדקה חשיבות המאפיינים במודל Random Forest.\n\n")
    f.write("עשרת המאפיינים החשובים ביותר:\n\n")
    f.write(top_features.to_markdown(index=False))
    f.write("\n\n")

    f.write(
        "המאפיין החשוב ביותר הוא PageValues, ובפער משמעותי משאר המאפיינים. "
        "מאפיין זה מייצג את הערך של הדפים שבהם המשתמש ביקר, ולכן הגיוני שהוא יהיה קשור מאוד לסיכוי לרכישה.\n\n"
    )

    f.write(
        "בנוסף, ExitRates ו-BounceRates נמצאו כמאפיינים חשובים. מאפיינים אלו קשורים להתנהגות יציאה מהאתר, "
        "ולכן הם יכולים להעיד על רמת עניין נמוכה או גבוהה של המשתמש. "
        "גם ProductRelated ו-ProductRelated_Duration היו חשובים, כלומר מספר דפי המוצר ומשך השהייה בהם "
        "קשורים לסיכוי שהביקור יסתיים ברכישה.\n\n"
    )

    f.write("## 10. מסקנות\n")
    f.write(
        "בפרויקט נמצא כי ניתן לחזות כוונת רכישה של מבקרים באתר קניות אונליין באמצעות מודלים של למידת מכונה. "
        "מבין המודלים שנבדקו, Random Forest נתן את הביצועים הטובים ביותר לפי מדד F1.\n\n"
    )

    f.write(
        "בנוסף, נמצא כי חוסר האיזון בדאטה משפיע מאוד על הערכת המודלים. "
        "מודל פשוט שמנבא תמיד אי רכישה יכול לקבל Accuracy גבוה, אך בפועל הוא אינו שימושי משום שאינו מזהה רכישות. "
        "לכן חשוב להשתמש במדדים נוספים כמו Precision, Recall ו-F1.\n\n"
    )

    f.write(
        "מבחינת מאפיינים, PageValues נמצא כמאפיין החשוב ביותר לחיזוי רכישה. "
        "גם מאפיינים המתארים את דפי המוצר ואת שיעורי היציאה מהאתר נמצאו משמעותיים. "
        "מסקנה זו מחזקת את ההבנה שהתנהגות המשתמש באתר יכולה לספק מידע חשוב על הסיכוי שיבצע רכישה.\n\n"
    )

    f.write("## 11. קבצים וגרפים שנוצרו\n")
    f.write("במהלך הפרויקט נוצרו מספר קבצים חשובים:\n\n")
    f.write("- results/model_comparison.csv\n")
    f.write("- results/imbalance_comparison.csv\n")
    f.write("- results/random_forest_feature_importance.csv\n")
    f.write("- results/revenue_distribution.png\n")
    f.write("- results/model_comparison_f1.png\n")
    f.write("- results/model_comparison_roc_auc.png\n")
    f.write("- results/random_forest_top_features.png\n")
    f.write("- results/imbalance_comparison_recall.png\n\n")

print("Final report created successfully.")
print("Saved file: report/final_report_hebrew.md")
