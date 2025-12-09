# ITRI_WindTurbine_AD
Anomaly Detection for wind turbines using spectrograms and acoustic images by a spatial-temporal convolutional autoencoder.

### Training configurations
For `Acoustic Images`, we use a **Spatial-Temporal Convolutional Autoencoder**: 
- Num Epochs: 20
- Batch size: 16
- Learning rate: 0.003
- Window size: 20

For `Spectrograms`, we use a vanilla **Convolutional Autoencoder**:
- Num Epochs: 150
- Batch size: 128
- Learning rate: 0.003

### Experimental Results
**NYCU_Small**
- Acoustic Images *(Threshold: mean + 5\*std)*

| | Low-In | Medium-In | Medium-Out | High-Out |
| :-: | :-: | :-: | :-: | :-: |
| Window=10 | 83.47% | 99.17% | 82.44% | 100% |
| Window=20 | 100% | 100% | 100% | 100% |
| Window=30 | 100% | 100% | 100% | 100% |


**ITRI_Small**
- Acoustic Images *(Threshold w=10: mean + 5\*std; w=20, 30: mean + 10\*std)*

|  | Anomaly|
| :-: | :-: |
| Window=10 | 99.65% |
| Window=20 | 99.98% |
| Window=30 | 99.98% |


- Spectrogram *(Threshold: mean + 0.5\*std)*

| | 0db | 6db | 12db |
| :-: | :-: | :-: | :-: |
| Accuracy | 95.56% | 100% | 100% |


**ITRI_Big**
- Spectrogram *(Threshold 0db, 6db: mean + 1\*std; 12db: mean + 0.5\*std)*

| | 0db | 6db | 12db |
| :-: | :-: | :-: | :-: |
| Accuracy | 85% | 85% | 95% |

| | 150 | 300 | 500m |
| :-: | :-: | :-: | :-: |
| Accuracy | 69% | 64% | 61% |
