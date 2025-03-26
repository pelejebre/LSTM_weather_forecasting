'''INDICE
train_val_test_split        --> Divide una serie temporal en conjuntos de entrenamiento, validación y prueba
create_dataset_supervised   --> Crea un conjunto de datos supervisado a partir de una serie temporal
scale_data                  --> Escala los datos utilizando MinMaxScaler
root_mean_squared_error     --> calcula el error cuadrático medio (RMSE)
r_squared                   --> Calcula el coeficiente de determinación (R²)
accuracy_threshold          --> Calcula el porcentaje de predicciones dentro de un umbral
'''

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import torch

def train_val_test_split(serie, tr_size=0.8, vl_size=0.1, ts_size=0.1):
    """
    Divide una serie temporal en conjuntos de entrenamiento, validación y prueba.
    
    Args:
        serie (pd.Series): Serie temporal a dividir.
        tr_size (float): Proporción del conjunto de entrenamiento.
        vl_size (float): Proporción del conjunto de validación.
        ts_size (float): Proporción del conjunto de prueba.
        
    Returns:
        tuple: Conjuntos de entrenamiento, validación y prueba.
    """
    N = serie.shape[0]
    # Calcular los índices para la división
    Ntrain = int(N * tr_size)   # Número de muestras de entrenamiento
    Nval = int(N * vl_size)     # Número de muestras de validación
    Ntest = N - Ntrain - Nval   # Número de muestras de prueba
    
    # Dividir la serie en conjuntos
    # Que en el caso de Forecasting, deben ser correlativos
    train = serie[0:Ntrain]
    val = serie[Ntrain:Ntrain+Nval]
    test = serie[Ntrain+Nval:]
    
    return train, val, test


# Función para crear los lotes/secuencias de datos conforme a la longitud de la ventana
# y a la longitud de la predicción
def create_dataset_supervised(array, input_length, output_length):
    """
    Crea un conjunto de datos supervisado a partir de una serie temporal.
    
    Args:
        array: arreglo numpy de tamaño N x feactures siendo N el número de muestras y features el número de características.
        input_length: longitud de la ventana de entrada.
        output_length: longitud de la ventana de salida.        
    Returns:
        X: conjunto de datos de entrada.
        y: conjunto de datos de salida.        
    """
    X, y = [], []
    shape = array.shape
    if len(shape) == 1:                    # Si el array es univariado (nuestro caso)
        fils, cols = array.shape[0], 1
        array = array.reshape(fils, cols)
    else:                                   # Si el array es multivariado
        fils, cols = array.shape 
    
    for i in range(fils-input_length-output_length):
        X.append(array[i:i+input_length,0:cols])
        y.append(array[i+input_length:i+input_length+output_length,-1].reshape(output_length,1))
        
    # Convertir a numpy array
    X = np.array(X)
    y = np.array(y)
    
    return X, y

# Función para escalar los datos de entrada y salida
def scale_data(data_input):
    """
    Escala los datos utilizando MinMaxScaler.
    
    Args:
        data_input: diccionario con los dataset de entrada y salida del modelo.
        (data_input = {'x_tr': x_tr, 'y_tr': y_tr, 'x_vl': x_vl, 'y_vl': y_vl, 'x_ts': x_ts, 'y_ts': y_ts})
        
    Returns:
        data_scaled: diccionario con los datasets de entrada y salida escalados
        scaler: objeto MinMaxScaler utilizado para la transformación.
    """
    Nfeatures = data_input['x_tr'].shape[2]
    
    # Generamos un listado de escaladores para cada feature
    scalers = [MinMaxScaler(feature_range=(-1, 1)) for i in range(Nfeatures)]
    
    # Inicializamos los arreglos para los datos escalados
    x_tr_s = np.zeros(data_input['x_tr'].shape)
    x_vl_s = np.zeros(data_input['x_vl'].shape)
    x_ts_s = np.zeros(data_input['x_ts'].shape)
    y_tr_s = np.zeros(data_input['y_tr'].shape)
    y_vl_s = np.zeros(data_input['y_vl'].shape)
    y_ts_s = np.zeros(data_input['y_ts'].shape)
        
    # Escalamos los arreglos de entrada (X)
    for i in range(Nfeatures):
        # Combinar todos los datos de entrada para ajustar el escalador
        # Primero aplanamos cada conjunto para poder concatenarlos
        x_tr_flat = data_input['x_tr'][:, :, i].reshape(-1, 1)
        x_vl_flat = data_input['x_vl'][:, :, i].reshape(-1, 1)
        x_ts_flat = data_input['x_ts'][:, :, i].reshape(-1, 1)
        
        # Combinamos todos los datos
        combined_data = np.vstack([x_tr_flat, x_vl_flat, x_ts_flat])
        
        # Ajustamos el escalador con todos los datos combinados
        scalers[i].fit(combined_data)
        
        # Escalamos cada conjunto por separado
        x_tr_s[:, :, i] = scalers[i].transform(x_tr_flat).reshape(data_input['x_tr'].shape[0], data_input['x_tr'].shape[1])
        x_vl_s[:, :, i] = scalers[i].transform(x_vl_flat).reshape(data_input['x_vl'].shape[0], data_input['x_vl'].shape[1])
        x_ts_s[:, :, i] = scalers[i].transform(x_ts_flat).reshape(data_input['x_ts'].shape[0], data_input['x_ts'].shape[1])
    
    # Escalamos los datos de salida (y)
    # Lo mismo que antes, pero para los datos de salida
    y_tr_flat = data_input['y_tr'][:, :, 0].reshape(-1, 1)
    y_vl_flat = data_input['y_vl'][:, :, 0].reshape(-1, 1)
    y_ts_flat = data_input['y_ts'][:, :, 0].reshape(-1, 1)
    
    combined_output = np.vstack([y_tr_flat, y_vl_flat, y_ts_flat])
    
    # Creamos un escalador para la salida
    output_scaler = MinMaxScaler(feature_range=(-1, 1))
    output_scaler.fit(combined_output)
    
    # Escalamos cada conjunto por separado
    y_tr_s[:, :, 0] = output_scaler.transform(y_tr_flat).reshape(data_input['y_tr'].shape[0], data_input['y_tr'].shape[1])
    y_vl_s[:, :, 0] = output_scaler.transform(y_vl_flat).reshape(data_input['y_vl'].shape[0], data_input['y_vl'].shape[1])
    y_ts_s[:, :, 0] = output_scaler.transform(y_ts_flat).reshape(data_input['y_ts'].shape[0], data_input['y_ts'].shape[1])
    
    # Añadimos el escalador de salida a la lista de escaladores
    scalers.append(output_scaler)
    
    # Construimos el diccionario de datos escalados
    data_scaled = {
        'x_tr_s': x_tr_s, 'y_tr_s': y_tr_s,
        'x_vl_s': x_vl_s, 'y_vl_s': y_vl_s,
        'x_ts_s': x_ts_s, 'y_ts_s': y_ts_s,        
    }
    
    return data_scaled, scalers[0]


# Deffinimos la función de pérdida RMSE personalizada
def root_mean_squared_error(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))

# Función para calcular R² (coeficiente de determinación)
def r_squared(y_pred, y_true):
    # Convertir tensores a numpy si es necesario
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().detach().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().detach().numpy()
    
    # Reshape para asegurar compatibilidad con sklearn
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    
    return r2_score(y_true, y_pred)

# Función para calcular "accuracy" basada en umbral
def accuracy_threshold(y_pred, y_true, threshold=0.1):
    """
    Calcula el porcentaje de predicciones que están dentro de un umbral
    del valor real (por ejemplo, dentro del 10% o 0.1 en escala normalizada)
    """
    # Convertir tensores a numpy si es necesario
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().detach().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().detach().numpy()
    
    # Reshape para trabajar con arrays unidimensionales
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    
    # Calcular error absoluto
    abs_error = np.abs(y_pred - y_true)
    
    # Contar predicciones dentro del umbral
    within_threshold = (abs_error <= threshold).sum()
    
    # Calcular porcentaje
    accuracy = within_threshold / len(y_true)
    
    return accuracy


def eval_rmse(model, x, y):
    model.eval()
    with torch.no_grad():
        pred = model(x)
        mse = mse_loss(pred, y)
        rmse = torch.sqrt(mse)
    return rmse.item()


def predictions(x, model, scaler, device=None):
    '''Genera la predicción de OUTPUT_LENGTH instantes
    de tiempo a futuro con el modelo entrenado.

    Entrada:
    - x: batch (o batches) de datos para ingresar al modelo
        (tamaño: BATCHES X INPUT_LENGTH X FEATURES)
    - model: Red LSTM entrenada
    - scaler: escalador (requerido para llevar la predicción a la escala original)
    - device: dispositivo donde ejecutar el modelo (cpu o cuda)

    Salida:
    - y_pred: la predicción en la escala original (tamaño: BATCHES X OUTPUT_LENGTH)
    '''
    # Si no se especifica dispositivo, usar CPU
    if device is None:
        device = torch.device('cpu')
    
    # Convertir a tensor de PyTorch si aún no lo es
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32).to(device)
    else:
        x = x.to(device)
    
    # Modo evaluación y no calcular gradientes para inferencia
    model.eval()
    with torch.no_grad():
        # Calcular predicción escalada en el rango usado para entrenar
        y_pred_s = model(x)
        
        # Convertir a numpy para usar con scaler
        y_pred_s = y_pred_s.cpu().numpy()
        
        # Reshape para que sea un array 2D (samples, features)
        original_shape = y_pred_s.shape
        y_pred_s_reshaped = y_pred_s.reshape(-1, original_shape[-1])
        
        # Llevar la predicción a la escala original
        y_pred_inverse = scaler.inverse_transform(y_pred_s_reshaped)
        
        # Restaurar la forma original si es necesario
        y_pred = y_pred_inverse.reshape(original_shape)
    
    return y_pred.flatten()