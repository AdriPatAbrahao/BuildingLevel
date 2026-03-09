import json
from pathlib import Path
import numpy as np

exp_dir = Path("outputs/experiments/20251128-183159_Treino_com_2000_amostras/metrics")

print("=" * 80)
print("LEITURA CORRETA DO CLASSIFICADOR - 2000 AMOSTRAS")
print("=" * 80)

# ✅ LER O ARQUIVO CORRETO
print("\n[1] CLASSIFIER_TEST.JSON (dados reais de teste)")
print("-" * 80)
with open(exp_dir / "classifier_test.json") as f:
    clf_test = json.load(f)

print(f"  Accuracy:  {clf_test.get('accuracy', 'N/A')}")
print(f"  Precision: {clf_test.get('precision', 'N/A')}")
print(f"  Recall:    {clf_test.get('recall', 'N/A')}")
print(f"  F1-Score:  {clf_test.get('f1', 'N/A')}")

# MATRIZ DE CONFUSÃO
if 'confusion_matrix' in clf_test:
    cm = clf_test['confusion_matrix']
    print(f"\n  Confusion Matrix:")
    print(f"    TN (Válidos bem classificados):     {cm[0][0]}")
    print(f"    FP (Válidos como inválidos):        {cm[0][1]}")
    print(f"    FN (Inválidos como válidos):        {cm[1][0]}")
    print(f"    TP (Inválidos bem classificados):   {cm[1][1]}")
    
    total = sum(sum(row) for row in cm)
    print(f"    Total: {total}")
    
    if cm[1][1] > 0:
        print(f"\n  ✓ O classificador SIM detecta soluções inválidas!")
        print(f"    Detectou {cm[1][1]} de {cm[1][1] + cm[1][0]} inválidos (Recall = {clf_test.get('recall', 0):.2%})")
    else:
        print(f"\n  ✗ O classificador NÃO detecta inválidos")

# ✅ LER TAMBÉM CLASSIFIER.JSON (treino)
print("\n[2] CLASSIFIER.JSON (dados de treino)")
print("-" * 80)
with open(exp_dir / "classifier.json") as f:
    classifier = json.load(f)

print(f"  Accuracy:  {classifier.get('accuracy', 'N/A')}")
print(f"  Precision: {classifier.get('precision', 'N/A')}")
print(f"  Recall:    {classifier.get('recall', 'N/A')}")
print(f"  F1-Score:  {classifier.get('f1_score', 'N/A')}")

# ✅ COMPARAR OS DOIS
print("\n[3] COMPARAÇÃO TREINO vs TESTE")
print("-" * 80)
print(f"  {'Métrica':<15} {'Treino':<15} {'Teste':<15} {'Diferença':<15}")
print(f"  {'-'*60}")

metrics = ['accuracy', 'precision', 'recall', 'f1_score']
for metric in metrics:
    train_val = classifier.get(metric, 0) if classifier.get(metric) else 0
    test_val = clf_test.get(metric if metric != 'f1_score' else 'f1', 0) if clf_test.get(metric if metric != 'f1_score' else 'f1') else 0
    diff = test_val - train_val
    
    print(f"  {metric:<15} {train_val:<15.4f} {test_val:<15.4f} {diff:<15.4f}")

# ✅ ROC CURVE
print("\n[4] CURVA ROC")
print("-" * 80)
with open(exp_dir / "roc_curve.json") as f:
    roc = json.load(f)

print(f"  AUC Score: {roc.get('auc', 'N/A')}")

# ✅ THRESHOLD CALIBRADO
print("\n[5] THRESHOLD CALIBRADO")
print("-" * 80)
with open(exp_dir / "validity_threshold.json") as f:
    threshold = json.load(f)

print(f"  Threshold Ótimo: {threshold.get('threshold', 'N/A')}")
print(f"  Método: {threshold.get('method', 'N/A')}")
print(f"  J-Score: {threshold.get('j_score', 'N/A')}")

print("\n" + "=" * 80)
print("CONCLUSÃO")
print("=" * 80)

# ANÁLISE FINAL
print("\n✓ O CLASSIFICADOR FUNCIONA?")
if clf_test.get('recall', 0) > 0:
    print(f"  SIM! Recall = {clf_test.get('recall', 0):.2%}")
    print(f"  Ele detecta {cm[1][1]} inválidos corretamente")
else:
    print(f"  NÃO. Recall = 0 (nunca detecta inválidos)")

print(f"\n✓ ESTÁ BALANCEADO?")
if clf_test.get('precision', 0) > 0.5 and clf_test.get('recall', 0) > 0.5:
    print(f"  SIM! Precision={clf_test.get('precision', 0):.2%}, Recall={clf_test.get('recall', 0):.2%}")
else:
    print(f"  NÃO. Precision={clf_test.get('precision', 0):.2%}, Recall={clf_test.get('recall', 0):.2%}")

print("\n" + "=" * 80)