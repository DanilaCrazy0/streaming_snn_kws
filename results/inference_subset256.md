# Доп. инференс-анализ (подвыборка val, 256 примеров, CPU)

Это **не** основные цифры диплома. Дипломные частоты спайков и потоковые метрики посчитаны на полном val/test и лежат в `thesis_report.json` / README.

Ниже — отчёт по 256 примерам валидации (нужны `.pkl` чекпоинты, в этот репозиторий веса не входят).

# Spike-Fusion inference analysis (ORCH_EXP_SPEC)

Ограничение: **без переобучения** — только загрузка `.pkl` и инференс.

- Device: `cpu`
- Val subset: **256** examples
- Batch size: 8

## Чекпоинты

- **best_rnn** (Best overall / RNN): `notebooks/artifacts_grid_search_sumfusion/fast_learning_rnn_sumfusion/bonus_phase3_p3_default_nfft512_hop64_k2_R3_H256/fast_k2_r3_H256_p3_default_nfft512_hop64_loss_final_prefix_consistency.pkl` (reported test=0.9268)
- **best_layered** (Best layered): `notebooks/artifacts_grid_search_sumfusion/fast_learning_sumfusion/final_k2_L2_H256_nfft256_hop64_p7/fast_k2_p1_L2_H256_p7_full_0_2_2_5_5_8_nfft256_hop64_loss_final_prefix_consistency.pkl` (reported test=0.9257)

## E1. Спайковая активность

### rnn
- `fullband`: **0.3459**
- `0_1_khz`: **0.3482**
- `1_4_khz`: **0.2976**
- `4_8_khz`: **0.3089**
- `fusion`: **0.3908**

### layered
- `fullband`: **0.2944**
- `0_2_khz`: **0.3050**
- `2_5_khz`: **0.3203**
- `5_8_khz`: **0.3638**
- `fusion`: **0.3804**

## E2. Мембранные потенциалы

- RNN mean(c|fire)=1.4168144464492798, mean(c|silent)=-1.734066128730774
- Layered mean(c|fire)=1.9467450380325317, mean(c|silent)=-2.6423263549804688

## E3a. State reset ablation (ключевой)

### rnn
- Streaming: final=0.9219, prefix=0.5726, ttc=79.67410714285714
- Reset each frame: final=0.2070, prefix=0.1531, ttc=69.77903225806452
- Δacc (streaming−reset) = **0.7148**
- KL(streaming∥reset) softmax traj = 1.5514137744903564

### layered
- Streaming: final=0.9297, prefix=0.5656, ttc=79.13578869047619
- Reset each frame: final=0.1562, prefix=0.1117, ttc=77.35520833333334
- Δacc (streaming−reset) = **0.7734**
- KL(streaming∥reset) softmax traj = 1.8296743631362915

## E3b. Frozen-input continuation

- Baseline final: 0.921875
- By freeze fraction: {"0.25": {"final_accuracy": 0.1953125, "prefix_accuracy": 0.19270312305889092}, "0.5": {"final_accuracy": 0.640625, "prefix_accuracy": 0.4728593761101365}, "0.75": {"final_accuracy": 0.8828125, "prefix_accuracy": 0.5680312486365438}}

## E3c. Prefix corruption (first 25% zeroed)

- Streaming final=0.921875, corrupted final=0.80859375, Δ=0.11328125

## E4. STFT / presets (из best_summary.json)

- Всего прогонов: 65
- Интерпретация STFT: hop=64 (4 ms) dominates; n_fft=512 preferred for RNN, n_fft=256 competitive for layered.
- Интерпретация presets: p3_default (4 branches) remains strongest for RNN; layered gains slightly from p7 band edges.
- Phase2 top-3:
  - bonus_phase2_nfft512_hop64_k2_R3_H256: test=0.9268 (n_fft=512, hop=64)
  - phase2_nfft512_hop64_k2_R2_H256: test=0.9264 (n_fft=512, hop=64)
  - phase2_nfft256_hop64_k2_R2_H256: test=0.9229 (n_fft=256, hop=64)

## E5. Prediction trajectory

- time_to_first_correct_mean: 79.91056910569105
- stability_after_hit_mean: 0.861396594954372

## Фигуры

- `fig_inf_spike_rates_branches.png` (+ копия в `figures/`)
- `fig_inf_membrane_dynamics.png` (+ копия в `figures/`)
- `fig_inf_memory_ablation.png` (+ копия в `figures/`)
- `fig_inf_frozen_input.png` (+ копия в `figures/`)
- `fig_inf_trajectory_probe.png` (+ копия в `figures/`)
- `fig_inf_branch_rate_correlation.png` (+ копия в `figures/`)
