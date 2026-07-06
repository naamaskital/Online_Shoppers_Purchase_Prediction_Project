import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("report", exist_ok=True)
os.makedirs("results", exist_ok=True)

model_comparison = pd.read_csv("results/model_comparison.csv")
imbalance_comparison = pd.read_csv("results/imbalance_comparison.csv")
feature_importance = pd.read_csv("results/random_forest_feature_importance.csv")

best_model = model_comparison.sort_values(by="F1", ascending=False).iloc[0]
top_features = feature_importance.head(10)

# Plot model comparison by F1
plt.figure(figsize=(10, 5))
plt.bar(model_comparison["Model"], model_comparison["F1"])
plt.title("Model Comparison by F1 Score")
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("results/model_comparison_f1.png")
plt.close()

# Plot model comparison by ROC AUC
plt.figure(figsize=(10, 5))
plt.bar(model_comparison["Model"], model_comparison["ROC_AUC"])
plt.title("Model Comparison by ROC AUC")
plt.xlabel("Model")
plt.ylabel("ROC AUC")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("results/model_comparison_roc_auc.png")
plt.close()

# Plot imbalance comparison by Recall
plt.figure(figsize=(12, 6))
plt.bar(imbalance_comparison["Model"], imbalance_comparison["Recall"])
plt.title("Effect of Class Imbalance on Recall")
plt.xlabel("Model")
plt.ylabel("Recall")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("results/imbalance_comparison_recall.png")
plt.close()

# Create Hebrew report draft
with open("report/report_summary_hebrew.md", "w", encoding="utf-8") as f:
    f.write("# סיכום תוצאות הפרויקט\n\n")

    f.write("## מטרת הפרויקט\n")
    f.write(
        "מטרת הפרויקט היא לחזות האם ביקור של משתמש באתר קניות אונליין "
        "יסתיים ברכישה או לא, לפי מאפיינים המתארים את התנהגות המשתמש באתר.\n\n"
    )

    f.write("## חוסר איזון בדאטה\n")
    f.write(
        "בדאטה קיימים 12,330 ביקורים. מתוכם 10,422 ביקורים לא הסתיימו ברכישה "
        "ו-1,908 ביקורים הסתיימו ברכישה. כלומר, רק כ-15.47% מהביקורים הסתיימו ברכישה.\n\n"
    )
    f.write(
        "מצב זה יוצר בעיית חוסר איזון בין המחלקות. לכן Accuracy לבדו אינו מדד מספיק, "
        "משום שמודל יכול לקבל Accuracy גבוה גם אם הוא כמעט תמיד מנבא שאין רכישה.\n\n"
    )

    f.write("## השוואת מודלים\n")
    f.write("המודלים שנבדקו בפרויקט הם Logistic Regression, KNN, Decision Tree, Random Forest ו-SVM.\n\n")
    f.write("טבלת התוצאות:\n\n")
    f.write(model_comparison.to_markdown(index=False))
    f.write("\n\n")

    f.write("## המודל הטוב ביותר\n")
    f.write(
        f"המודל הטוב ביותר לפי מדד F1 הוא {best_model['Model']}. "
        f"המודל הגיע ל-Accuracy של {best_model['Accuracy']:.4f}, "
        f"Precision של {best_model['Precision']:.4f}, "
        f"Recall של {best_model['Recall']:.4f}, "
        f"F1 של {best_model['F1']:.4f}, "
        f"and ROC AUC של {best_model['ROC_AUC']:.4f}.\n\n"
    )
    f.write(
        "Random Forest הצליח לתת איזון טוב בין Precision ו-Recall, ולכן הוא מתאים במיוחד "
        "לבעיה שבה חשוב לזהות רכישות, אך גם לא לייצר יותר מדי תחזיות שגויות.\n\n"
    )

    f.write("## השפעת חוסר האיזון\n")
    f.write(
        "כדי לבדוק את השפעת חוסר האיזון, הושווה מודל Dummy שמנבא תמיד את המחלקה הנפוצה ביותר. "
        "מודל זה קיבל Accuracy של כ-0.845, אך Precision, Recall ו-F1 שלו עבור מחלקת הרכישה היו 0. "
        "מכאן ניתן לראות ש-Accuracy לבדו עלול להטעות.\n\n"
    )
    f.write(
        "בנוסף, כאשר השתמשנו ב-class_weight='balanced', ניתן לראות שבחלק מהמודלים ה-Recall השתפר משמעותית. "
        "לדוגמה, ב-Logistic Regression ה-Recall עלה מ-0.356 ל-0.749. "
        "כלומר, המודל המאוזן הצליח לזהות הרבה יותר ביקורים שהסתיימו ברכישה.\n\n"
    )

    f.write("## מאפיינים חשובים\n")
    f.write("לפי Random Forest, עשרת המאפיינים החשובים ביותר הם:\n\n")
    f.write(top_features.to_markdown(index=False))
    f.write("\n\n")

    f.write(
        "המאפיין החשוב ביותר הוא PageValues, בפער גדול משאר המאפיינים. "
        "בנוסף, גם ExitRates, ProductRelated_Duration, ProductRelated ו-BounceRates נמצאו כמאפיינים חשובים. "
        "מכאן ניתן להסיק שהתנהגות המשתמש באתר, במיוחד צפייה בדפי מוצר, זמן שהייה וערך הדפים, "
        "קשורה מאוד לסיכוי שהביקור יסתיים ברכישה.\n\n"
    )

    f.write("## מסקנה\n")
    f.write(
        "ניתן לחזות כוונת רכישה של מבקרים באתר קניות אונליין באמצעות מודלים של למידת מכונה. "
        "מבין המודלים שנבדקו, Random Forest נתן את הביצועים הטובים ביותר. "
        "בנוסף, נמצא שחוסר האיזון בדאטה משפיע מאוד על הערכת המודלים, ולכן חשוב להשתמש במדדים כמו Precision, Recall ו-F1 ולא רק ב-Accuracy.\n"
    )

print("Report summary created successfully.")
print("Saved file: report/report_summary_hebrew.md")
print("Saved plots:")
print("- results/model_comparison_f1.png")
print("- results/model_comparison_roc_auc.png")
print("- results/imbalance_comparison_recall.png")
