import optuna
import json
import os
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.train_optimization import train_model, DEFAULT_CONFIG # Importa do seu script adaptado


# Configurações do Pipeline
N_TRIALS_CYCLE_1 = 15  # Testes rápidos de janela
N_TRIALS_CYCLE_2 = 50  # Testes pesados de arquitetura (O mais importante)
N_TRIALS_CYCLE_3 = 20  # Ajuste fino

RESULTS_DIR = "optimization_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

final_best_params = {} # Acumula os vencedores de cada etapa

def save_stage_results(stage_name, study, params):
    """Salva os resultados parciais em JSON"""
    path = os.path.join(RESULTS_DIR, f"{stage_name}_best.json")
    with open(path, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"✅ Resultados de {stage_name} salvos em {path}")

# --- CICLO 1: DADOS ---
def objective_cycle_1(trial):
    # Fixamos o resto, variamos apenas a janela
    params = {
        'input_seq_len': trial.suggest_categorical('input_seq_len', [12, 24, 48, 72, 96]),
        # Poderia incluir 'use_mask' aqui se quisesse
    }
    # Roda o treino passando os params
    return train_model(params, trial)

# --- CICLO 2: ARQUITETURA (Usa o vencedor do Ciclo 1) ---
def objective_cycle_2(trial):
    # Recupera o vencedor do ciclo anterior
    base_params = final_best_params.copy()
    
    # 1. Definir a Arquitetura (ID) em variável local
    # O Optuna guarda isso automaticamente no histórico do trial
    arch_id = trial.suggest_categorical('hidden_sizes_id', ['p', 'm', 'g', 'gg'])
    
    # 2. Mapa de Tradução (ID -> Lista)
    arch_map = {
        'p': [64],
        'm': [128],
        'g': [128, 64],
        'gg': [256, 128]
    }
    
    # 3. Montar os parâmetros finais
    # Repare que passamos direto a LISTA de hidden_sizes, não o ID
    base_params.update({
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'hidden_sizes': arch_map[arch_id], # <--- Aqui estava o erro, agora vai direto
        'cell_type': trial.suggest_categorical('cell_type', ['gru', 'lstm']),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128])
    })
    
    # Não precisamos mais deletar chaves ou ler do dicionário
    return train_model(base_params, trial)

# --- CICLO 3: FINE TUNING (Usa vencedores do 1 e 2) ---
def objective_cycle_3(trial):
    base_params = final_best_params.copy()
    
    base_params.update({
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        # Se quiser testar weight_decay:
        # 'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    })
    
    return train_model(base_params, trial)

def main():
    print("🚀 Iniciando Pipeline de Otimização em 3 Estágios")
    
    # --- EXECUÇÃO CICLO 1 ---
    print("\n--- [Ciclo 1/3] Otimizando Janela de Tempo ---")
    study1 = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
    study1.optimize(objective_cycle_1, n_trials=N_TRIALS_CYCLE_1)
    
    print(f"🏆 Melhor Ciclo 1: {study1.best_params}")
    final_best_params.update(study1.best_params)
    save_stage_results("cycle_1_data", study1, study1.best_params)

    # --- EXECUÇÃO CICLO 2 ---
    print("\n--- [Ciclo 2/3] Otimizando Arquitetura e LR ---")
    study2 = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
    study2.optimize(objective_cycle_2, n_trials=N_TRIALS_CYCLE_2)
    
    # Tratamento especial para o 'hidden_sizes' que usamos mapeamento
    best_c2 = study2.best_params
    arch_map = {'p': [64], 'm': [128], 'g': [128, 64], 'gg': [256, 128]}
    real_hidden = arch_map[best_c2['hidden_sizes_id']]
    
    # Atualiza dicionário final
    final_best_params.update(best_c2)
    final_best_params['hidden_sizes'] = real_hidden
    del final_best_params['hidden_sizes_id'] # Limpa auxiliar
    
    print(f"🏆 Melhor Ciclo 2: {final_best_params}")
    save_stage_results("cycle_2_arch", study2, final_best_params)

    # --- EXECUÇÃO CICLO 3 ---
    print("\n--- [Ciclo 3/3] Ajuste Fino (Dropout) ---")
    study3 = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
    study3.optimize(objective_cycle_3, n_trials=N_TRIALS_CYCLE_3)
    
    final_best_params.update(study3.best_params)
    print(f"🏆 Melhor Ciclo 3: {study3.best_params}")
    save_stage_results("cycle_3_finetune", study3, final_best_params)

    # --- RELATÓRIO FINAL ---
    print("\n📝 Gerando Relatório Final...")
    report_path = os.path.join(RESULTS_DIR, "FINAL_REPORT.txt")
    with open(report_path, "w") as f:
        f.write("RELATÓRIO DE OTIMIZAÇÃO DE HIPERPARÂMETROS\n")
        f.write("==========================================\n\n")
        f.write(f"Data: {datetime.now()}\n")
        f.write(f"Estratégia: Optuna TPE (3 Estágios)\n\n")
        
        f.write("MELHOR CONFIGURAÇÃO ENCONTRADA:\n")
        json.dump(final_best_params, f, indent=4)
        
        f.write("\n\nHISTÓRICO:\n")
        f.write(f"Ciclo 1 (Janela): Melhor Loss = {study1.best_value:.6f}\n")
        f.write(f"Ciclo 2 (Arquit): Melhor Loss = {study2.best_value:.6f}\n")
        f.write(f"Ciclo 3 (Fino):   Melhor Loss = {study3.best_value:.6f}\n")
    
    print(f"✨ Processo concluído! Veja o relatório em: {report_path}")

if __name__ == "__main__":
    main()