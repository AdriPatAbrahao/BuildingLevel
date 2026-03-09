import json
import pandas as pd
import numpy as np
from pathlib import Path

def safe_float(val, default=0.0):
    """Converte valor para float, retornando default se for inválido."""
    try:
        return float(val) if val != 'N/A' else default
    except (TypeError, ValueError):
        return default

# Carrega os principais arquivos de métricas
exp_dir = Path("outputs/experiments/20251128-183159_Treino_com_2000_amostras/metrics")

# 1. RESUMO GERAL
print("=" * 80)
print("ANÁLISE DO TREINAMENTO COM 2000 AMOSTRAS")
print("=" * 80)

with open(exp_dir / "summary.json") as f:
    summary = json.load(f)

print("\n[1] RESUMO GERAL DO TREINAMENTO")
print("-" * 80)
print(f"  Amostras Treinadas: {summary.get('num_samples_trained', 'N/A')}")
print(f"  Amostras Teste:     {summary.get('num_test_samples', 'N/A')}")
print(f"  Dispositivo:        {summary.get('device', 'N/A')}")
print(f"  Taxa CPU:           {summary.get('resources_snapshot', {}).get('cpu_percent', 'N/A')}%")
print(f"  Memória (MB):       {summary.get('resources_snapshot', {}).get('memory_rss_mb', 'N/A')}")

# 2. MÉTRICAS FINAIS DO MODELO DE REGRESSÃO
print("\n[2] PERFORMANCE DO MODELO DE REGRESSÃO (AÇO)")
print("-" * 80)
final_metrics = summary.get('final_metrics', {}).get('steel', {})
print(f"  R² Score:           {safe_float(final_metrics.get('r2_score'), 0):.4f}")
print(f"  MAE (kgf):          {safe_float(final_metrics.get('mean_absolute_error_kgf'), 0):.2f}")
print(f"  RMSE (kgf):         {safe_float(final_metrics.get('rmse_kgf'), 0):.2f}")

residual_stats = final_metrics.get('residual_stats', {})
print(f"  Residual Stats:")
print(f"    Mean Abs Error:   {safe_float(residual_stats.get('mean_abs_error_kgf'), 0):.2f} kgf")
print(f"    Std Abs Error:    {safe_float(residual_stats.get('std_abs_error_kgf'), 0):.2f} kgf")
print(f"    Max Abs Error:    {safe_float(residual_stats.get('max_abs_error_kgf'), 0):.2f} kgf")

# 3. CLASSIFICADOR
print("\n[3] PERFORMANCE DO CLASSIFICADOR DE VALIDADE")
print("-" * 80)
with open(exp_dir / "classifier.json") as f:
    classifier = json.load(f)

print(f"  Accuracy:           {safe_float(classifier.get('accuracy'), 0):.4f}")
print(f"  Precision:          {safe_float(classifier.get('precision'), 0):.4f}")
print(f"  Recall:             {safe_float(classifier.get('recall'), 0):.4f}")
print(f"  F1-Score:           {safe_float(classifier.get('f1_score'), 0):.4f}")

# 4. THRESHOLD DE VALIDADE
print("\n[4] THRESHOLD DE VALIDADE CALIBRADO")
print("-" * 80)
with open(exp_dir / "validity_threshold.json") as f:
    threshold = json.load(f)

print(f"  Threshold Ótimo:    {safe_float(threshold.get('threshold'), 0.5):.4f}")
print(f"  Método:             {threshold.get('method', 'N/A')}")
print(f"  J-Score (Youden):   {safe_float(threshold.get('j_score'), 0):.4f}")

# 5. CURVA ROC
print("\n[5] ANÁLISE DA CURVA ROC")
print("-" * 80)
with open(exp_dir / "roc_curve.json") as f:
    roc = json.load(f)

auc = roc.get("auc", 0)
print(f"  AUC Score:          {safe_float(auc, 0):.4f}")

# 6. TIMINGS
print("\n[6] TEMPOS DE EXECUÇÃO")
print("-" * 80)
timings = summary.get('timings_detailed', {})
print(f"  Split dados (s):    {safe_float(timings.get('split_sec'), 0):.6f}")
print(f"  Scaling (s):        {safe_float(timings.get('scaling_sec'), 0):.6f}")
print(f"  Treino NN (s):      {safe_float(timings.get('train_nn_sec'), 0):.6f}")
print(f"  Treino Classifier (s): {safe_float(timings.get('train_classifier_sec'), 0):.6f}")

tqs_times = summary.get('tqs_phase_times_sec', {})
print(f"\n  TQS Modeling (s):   {safe_float(tqs_times.get('modeling'), 0):.2f}")
print(f"  TQS Execution (s):  {safe_float(tqs_times.get('execution'), 0):.2f}")

# 7. EVOLUÇÃO DO TREINAMENTO (EPOCHS)
print("\n[7] EVOLUÇÃO DO TREINAMENTO POR ÉPOCA")
print("-" * 80)
with open(exp_dir / "epochs.ndjson") as f:
    epochs = [json.loads(line) for line in f]

print(f"  Total de Épocas:    {len(epochs)}")
print(f"\n  Primeiras 5 épocas:")
for i, ep in enumerate(epochs[:5]):
    train_loss = safe_float(ep.get('train_loss'), 0)
    val_loss = safe_float(ep.get('val_loss'), 0)
    r2 = safe_float(ep.get('r2_score'), 0)
    print(f"    Época {i:2d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, r2={r2:.4f}")

print(f"\n  Últimas 5 épocas:")
for i, ep in enumerate(epochs[-5:], start=len(epochs)-5):
    train_loss = safe_float(ep.get('train_loss'), 0)
    val_loss = safe_float(ep.get('val_loss'), 0)
    r2 = safe_float(ep.get('r2_score'), 0)
    print(f"    Época {i:2d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, r2={r2:.4f}")

# Melhor época
if epochs:
    best_epoch = min(epochs, key=lambda x: safe_float(x.get('val_loss'), float('inf')))
    best_idx = epochs.index(best_epoch)
    print(f"\n  Melhor Época: {best_idx}")
    print(f"    train_loss: {safe_float(best_epoch.get('train_loss'), 0):.6f}")
    print(f"    val_loss:   {safe_float(best_epoch.get('val_loss'), 0):.6f}")
    print(f"    r2_score:   {safe_float(best_epoch.get('r2_score'), 0):.4f}")

# 8. CRITÉRIOS DE SUCESSO
print("\n[8] CRITÉRIOS DE SUCESSO")
print("-" * 80)
criteria = summary.get('criteria_status', {})
for criterion, status in criteria.items():
    status_symbol = "✓" if status else "✗"
    print(f"  {status_symbol} {criterion}: {status}")

# 9. IMPORTÂNCIA DAS FEATURES
print("\n[9] IMPORTÂNCIA DAS FEATURES")
print("-" * 80)
try:
    with open(exp_dir / "feature_importance.json") as f:
        feat_imp = json.load(f)
    
    sorted_features = sorted(feat_imp.items(), key=lambda x: abs(safe_float(x[1], 0)), reverse=True)
    print(f"  Top 10 Features mais Importantes:")
    for i, (feat, imp) in enumerate(sorted_features[:10], 1):
        print(f"    {i:2d}. {feat:<30} {safe_float(imp, 0):>10.6f}")
except FileNotFoundError:
    print("  Feature importance não disponível")

# 10. COMPARAÇÃO COM OUTROS TREINAMENTOS
print("\n[10] COMPARAÇÃO COM OUTROS TREINAMENTOS")
print("-" * 80)

experiments = {
    "2000 amostras":     "outputs/experiments/20251128-183159_Treino_com_2000_amostras/metrics/summary.json"
}

print(f"  {'Modelo':<25} {'R²':<10} {'MAE Aço':<12} {'Acc Class':<12} {'AUC ROC':<10}")
print(f"  {'-'*70}")

for name, path in experiments.items():
    try:
        with open(path) as f:
            data = json.load(f)
            r2 = safe_float(data.get('final_metrics', {}).get('steel', {}).get('r2_score'), 0)
            mae = safe_float(data.get('final_metrics', {}).get('steel', {}).get('mean_absolute_error_kgf'), 0)
            
            with open(Path(path).parent / "classifier.json") as cf:
                clf_data = json.load(cf)
                acc = safe_float(clf_data.get('accuracy'), 0)
            
            with open(Path(path).parent / "roc_curve.json") as rf:
                roc_data = json.load(rf)
                auc = safe_float(roc_data.get('auc'), 0)
            
            print(f"  {name:<25} {r2:<10.4f} {mae:<12.2f} {acc:<12.4f} {auc:<10.4f}")
    except FileNotFoundError as e:
        print(f"  {name:<25} Arquivo não encontrado")

print("\n" + "=" * 80)
print("RECOMENDAÇÕES BASEADAS NA ANÁLISE")
print("=" * 80)

# Análise de recomendações
r2 = safe_float(summary.get('final_metrics', {}).get('steel', {}).get('r2_score'), 0)
mae = safe_float(summary.get('final_metrics', {}).get('steel', {}).get('mean_absolute_error_kgf'), 0)
acc = safe_float(classifier.get('accuracy'), 0)

print("\n✓ PONTOS POSITIVOS:")
if acc > 0.80:
    print(f"  • Classificador de validade EXCELENTE (Accuracy={acc:.2%})")
if safe_float(roc.get('auc'), 0) > 0.90:
    print(f"  • ROC AUC muito bom (0.916) → alta capacidade de discriminação")
if summary.get('num_samples_trained', 0) >= 900:
    print(f"  • Volume de dados adequado ({summary.get('num_samples_trained')} amostras treinadas)")

print("\n✗ PONTOS A MELHORAR:")
if r2 < 0.70:
    print(f"  • R² do modelo de regressão baixo (0.677) → explica apenas 67.7% da variância")
if mae > 50:
    print(f"  • MAE alto ({mae:.2f} kgf) → erros médios significativos nas predições")

print("\n🎯 AÇÕES RECOMENDADAS:")
if r2 < 0.70:
    print(f"  1. Aumentar amostras de treinamento (tentar 3000-5000)")
    print(f"  2. Revisar features: pode haver features correlacionadas ou irrelevantes")
    print(f"  3. Ajustar arquitetura da rede: tentar mais camadas ou regularização")
    print(f"  4. Verificar qualidade dos dados TQS (outliers, erros)")

print("\n" + "=" * 80)

# LISTA DE FEATURES
print("\n[11] LISTA DE FEATURES UTILIZADAS")
print("-" * 80)
with open(exp_dir / "feature_names.json") as f:
    features = json.load(f)

print(f"  Total de features: {len(features)}")
print("\n  Features:")
for i, feat in enumerate(features, 1):
    print(f"    {i}. {feat}")

# Ler arquivo de teste do classificador
with open(exp_dir / "classifier_test.json") as f:
    clf_test = json.load(f)

print("Classificador Test Results:")
print(f"  Accuracy: {clf_test.get('accuracy', 'N/A')}")
print(f"  Precision: {clf_test.get('precision', 'N/A')}")
print(f"  Recall: {clf_test.get('recall', 'N/A')}")
print(f"  F1: {clf_test.get('f1', 'N/A')}")

# Verificar matriz de confusão
if 'confusion_matrix' in clf_test:
    cm = clf_test['confusion_matrix']
    print(f"\n  Confusion Matrix:")
    print(f"    TN (True Negative):  {cm[0][0]}")  # Válidos bem classificados
    print(f"    FP (False Positive): {cm[0][1]}")  # Válidos como inválidos
    print(f"    FN (False Negative): {cm[1][0]}")  # Inválidos como válidos 🔴
    print(f"    TP (True Positive):  {cm[1][1]}")  # Inválidos bem classificados
    
    if cm[1][1] == 0:
        print("\n  ⚠️  PROBLEMA: O classificador NUNCA previu a classe positiva (inválido)!")

# -----------------------------------
# LEITURA CORRETA DO CLASSIFICADOR
# -----------------------------------

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