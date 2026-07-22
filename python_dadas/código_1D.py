import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import librosa


#Parâmetros

L = 1.0          # comprimento (m)
T = 100.0        # tensão (N)
mu = 0.01        # densidade linear (kg/m)
c = np.sqrt(T/mu)
f1 = c / (2*L)   # frequência fundamental (Hz)
N_modos = 20     # quantos modos considerar
zeta = 0.01      # fator de amortecimento (adimensional)


audio_path = "../data/musica-1.mp3"   
y, fs = librosa.load(audio_path, sr=None, mono=True)

# ========== CORTAR SILÊNCIO ==========
y, _ = librosa.effects.trim(y, top_db=30)   # remove silêncio no início e fim

# Normalizar
y = y / np.max(np.abs(y))

N_amostras = len(y)
t_audio = np.arange(N_amostras) / fs

#FFT
X = fft(y)
freq = fftfreq(N_amostras, 1/fs)  
omega = 2 * np.pi * freq

# Processar cada modo
# Armazenar as respostas temporais de cada modo
respostas_modos = []

for n in range(1, N_modos+1):
    f_n = n * f1
    omega_n = 2 * np.pi * f_n
    
    # Função de Transferência do oscilador (modo n)
    H = 1 / ((omega_n**2 - omega**2) + 2j * zeta * omega_n * omega)
    
    # Multiplicação no domínio da frequência (a "mágica")
    Y_n = H * X
    
    # Resposta no tempo (parte real da IFFT)
    y_n = np.real(ifft(Y_n))
    respostas_modos.append(y_n)
    
    # (Opcional) imprimir força e intensidade do modo
    idx = np.argmin(np.abs(freq - f_n))
    forca = np.abs(X[idx])
    intensidade = np.abs(Y_n[idx])
    print(f"Modo {n:2d}: f={f_n:6.2f} Hz  | Força={forca:.4f}  | Intensidade={intensidade:.4f}")

# ============================================
# 5. Reconstruir a vibração da corda no espaço
# ============================================
x_espaco = np.linspace(0, L, 200)  # pontos ao longo da corda

# Escolha um instante de tempo para plotar (ex: t = 0.5 s)
t_plot = 1.5
idx_t = int(t_plot * fs)  # índice da amostra mais próxima

# Calcular o deslocamento u(x, t_plot) somando os modos
u = np.zeros_like(x_espaco)
for n, y_n in enumerate(respostas_modos, start=1):
    u += np.sin(n * np.pi * x_espaco / L) * y_n[idx_t]

# Plotar a forma da corda naquele instante
plt.figure(figsize=(10, 4))
plt.plot(x_espaco, u, linewidth=2)
plt.title(f'Corda vibrante em t = {t_plot:.2f} s')
plt.xlabel('Posição x (m)')
plt.ylabel('Deslocamento u(x,t)')
plt.grid(True)
plt.show()

###########################


import pandas as pd

# Salvar as respostas modais (amostras) em CSV
# Vamos salvar apenas uma subamostra para não gerar arquivo gigante
passo_salvar = 1000  # salva 1 a cada 1000 amostras
indices_salvar = np.arange(0, len(respostas_modos[0]), passo_salvar)

# Criar DataFrame com as amplitudes de cada modo nos instantes selecionados
df = pd.DataFrame()
df['tempo (s)'] = indices_salvar / fs
for n, y_n in enumerate(respostas_modos, start=1):
    df[f'modo_{n}'] = y_n[indices_salvar]

df.to_csv('respostas_modais.csv', index=False)
print("Dados modais salvos em 'respostas_modais.csv'")

# Salvar também a posição da corda em um instante específico (ex: t=0.5s)
t_plot = 0.5
idx_t = int(t_plot * fs)
u_plot = np.zeros_like(x_espaco)
for n, y_n in enumerate(respostas_modos, start=1):
    u_plot += np.sin(n * np.pi * x_espaco / L) * y_n[idx_t]

df_pos = pd.DataFrame({'x (m)': x_espaco, 'u (m)': u_plot})
df_pos.to_csv('corda_t_0.5s.csv', index=False)
print("Posição da corda em t=0.5s salva em 'corda_t_0.5s.csv'")

###########################

# ANIMACAO


import matplotlib
matplotlib.use('Agg')  # Backend sem janela (salva em arquivo)
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parâmetros
num_frames = 300
total_amostras = len(respostas_modos[0])
step = max(1, total_amostras // num_frames)
indices = np.arange(0, total_amostras, step)[:num_frames]

x_espaco = np.linspace(0, L, 100)

fig, ax = plt.subplots(figsize=(10,4))
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, L)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('Posição x (m)')
ax.set_ylabel('Deslocamento')
ax.grid(True)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    u_frame = np.zeros_like(x_espaco)
    for n, y_n in enumerate(respostas_modos, start=1):
        u_frame += np.sin(n * np.pi * x_espaco / L) * y_n[indices[i]]
    line.set_data(x_espaco, u_frame)
    ax.set_title(f't = {indices[i]/fs:.3f} s')
    return line,

ani = FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=50, blit=True)

# Salvar como vídeo (precisa de ffmpeg instalado)
ani.save('corda_vibrando.mp4', writer='ffmpeg', fps=20)
print("Vídeo salvo como 'corda_vibrando.mp4'")