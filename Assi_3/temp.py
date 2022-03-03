    "import re
    "import time
    "import random
    "import imageio
    "import numpy as np
    "import matplotlib.pyplot as plt
    "from collections import Counter
    "from matplotlib.figure import Figure
    "from sklearn.decomposition import PCA
    "from scipy.stats import multivariate_normal
    "from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    "import glob
    "# DFT matrix to caluclate 64 point DFT from a 400-D signal
    "K, N = 64, 400
    "F = np.zeros((K,N), np.complex64)
    "
    "for k in range(K):
    "    for n in range(N):
    "        F[k,n] =  np.exp(-2j *k * n * np.pi/N)
    "
    "
    "
    "
    "def spectrogram(x, win_len=400, hop_len=160, n_fft=256):
    "    '''
    "    spectrogram
    "    
    "    Inputs
    "    ------
    "    x: shape(N_f, ) - raw signal
    "    win_len: int - window length in frames
    "    hop_len: int - hop length in frames
    "    n_fft: int - number of fft points
    "    
    "    Returns
    "    -------
    "    xf: shape(N_f, 128) - spectrogram
    "    '''
    "    # calculate number of frames
    "    n_frames = 1 + np.int(np.floor((len(x) - win_len) / float(hop_len)))
    "    xf = np.zeros((n_frames, K), np.complex64)
    "    # perform sliding window operation
    "    for i in range(n_frames):
    "        z = x[i * hop_len: i * hop_len + win_len] * np.hamming(win_len)
    "        z2 = F @ z
    "        
    "        xf[i] = z2
    "    # obtain log of the absolute values of the features
    "    xf = np.log(np.abs(xf[:, :K // 2])+1e-8)                     
    "    return xf
    "
    "
    "def get_spec_frames(data_path):
    "    '''
    "    data_path: path to the .wav files
    "
    "    returns
    "    -----
    "    frames: shape(*,32) ---> each row is a feature vector
    "    '''
    "    path = data_path
    "    dir_list  = glob.glob(data_path + '/*.wav')
    "    #print(len(dir_list))
    "
    "    frames = []
    "
    "    for i in range(len(dir_list)):
    "        y,fs = librosa.load(dir_list[i],sr = None)
    "        y_spec = spectrogram(y)
    "        frames.extend(y_spec)
    "            
    "    frames = np.array(frames)
    "    return frames
    "\n"
    "class KMeans:
    "    def __init__(self, n_clusters=2, max_iter=1000):
    "        self.n_clusters = n_clusters
    "        self.max_iter = max_iter
    "        self.centers = None
    "        
    "    def dist(self, data_points, center):
    "        return np.sqrt(np.sum((data_points - center) ** 2, axis=1))
    "        
    "    def fit(self, data):
    "        # get initial centers from the samples itself.
    "        # better initializations are also available - have a look at kmeans++
    "        self.centers = data[
    "            np.random.choice(np.arange(len(data)), size=self.n_clusters)
    "        ]
    "        
    "        # kmeans algorithm
    "        for _ in range(self.max_iter):
    "            dist_matrix = np.zeros((len(data), self.n_clusters))
    "            for i in  range(self.n_clusters):
    "                dist_matrix[:, i] = self.dist(data, self.centers[i])
    "            cluster_assignment = dist_matrix.argmin(axis=1)
    "            # find new centers
    "            new_centers = np.zeros_like(self.centers)
    "            for i in range(self.n_clusters):
    "                new_centers[i] = np.mean(data[cluster_assignment == i], axis=0)
    "            # break if we have converged
    "            if np.allclose(new_centers, self.centers):
    "                self.centers = new_centers
    "                break
    "            self.centers = new_centers
    "        
    "    def predict(self, data):
    "        dist_matrix = np.zeros((len(data), self.n_clusters))
    "        for i in range(self.n_clusters):
    "            dist_matrix[:, i] = self.dist(data, self.centers[i])
    "        cluster_assignment = dist_matrix.argmin(axis=1)
    "class GMM:
    "    def __init__(
    "        self, 
    "        n_mixtures=2, 
    "        max_iter=100, 
    "        covar_type='full', 
    "        plot_progress=False, 
    "        p_iter=5,
    "        gif_name='EM_Progress.gif'
    "    ):
    "        self.n_mixtures = n_mixtures
    "        self.max_iter = max_iter
    "        self.alphas = np.ones(n_mixtures) / n_mixtures
    "        self.means = None
    "        self.covs = None
    "        self.covar_type = covar_type
    "        self.plot_progress = plot_progress
    "        self.p_iter = p_iter
    "        self.gif_name = gif_name
    "        self.log_likelihood_plot_list = None
    "        self.cluster_idx = None
    "        
    "    def p(self, data, mean, cov):
    "        dist = multivariate_normal(mean=mean, cov=cov)
    "        return dist.pdf(data)
    "    
    "    def predict(self, data):
    "        posteriors = np.zeros((len(data), self.n_mixtures))
    "        for i in range(self.n_mixtures):
    "            posteriors[:, i] = self.alphas[i] * self.p(data, self.means[i], self.covs[i])
    "        labels = np.argmax(posteriors, axis=1)
    "        self.cluster_idx = labels
    "        return labels
    "    
    "    def full_covar(self, data, resp):
    "        new_covs = np.zeros_like(self.covs)
    "        for i in range(self.n_mixtures):
    "            diff = data - self.means[i]
    "            new_covs[i] = np.dot(resp[:, i] * diff.T, diff) / resp[:, i].sum()
    "            # regularization term to keep the covariance matrix positive semi-definite
    "            new_covs[i].flat[::data.shape[1] + 1] += 1e-6
    "        return new_covs
    "            
    "    def diag_covar(self, data, resp):
    "        sqrd_data = np.dot(resp.T, data * data) / resp.sum(axis=0)[:, None]
    "        sqrd_means = self.means ** 2
    "        means = self.means * np.dot(resp.T, data) / resp.sum(axis=0)[:, None]
    "        diag_covs = (sqrd_data - 2 * means + sqrd_means + 1e-6)
    "        new_covs = [np.diag(cov) for cov in diag_covs]
    "        return new_covs
    "
    "    def get_log_likelihood(self, data):
    "        probs = np.zeros((len(data), self.n_mixtures))
    "        for i in range(self.n_mixtures):
    "            probs[:, i] = self.alphas[i] * self.p(data, self.means[i], self.covs[i])
    "
    "        ll_hood = np.sum(np.log(probs.sum(axis=1)))
    "
    "        return ll_hood
    "    
    "    def e_step(self, data):
    "        resp = np.zeros((len(data), self.n_mixtures))
    "        # find responsibility of each data point towards a Gaussian
    "        for i in range(self.n_mixtures):
    "            resp[:, i] = self.alphas[i] * self.p(data, self.means[i], self.covs[i])
    "            
    "        self.log_likelihood_plot_list.append(np.sum(np.log(resp.sum(axis=1)))) 
    "        # normalize the sum
    "        resp = resp / resp.sum(axis=1)[:, None]
    "        return resp
    "        
    "    def m_step(self, data, resp):        
    "        # re-estimation for alphas
    "        new_alphas = resp.mean(axis=0)
    "        
    "        # re-estimation for means
    "        new_means = np.zeros_like(self.means)
    "        for i in range(self.n_mixtures):
    "            new_means[i] = np.multiply(resp[:, i][:, None], data).sum(axis=0) / resp[:, i].sum()
    "        
    "        # re-estimation for covariance matrix
    "        if self.covar_type == 'full':
    "            new_covs = self.full_covar(data, resp)
    "        elif self.covar_type == 'diag':
    "            new_covs = self.diag_covar(data, resp)
    "        else:
    "            raise NotImplementedError()
    "            
    "        return new_alphas, new_means, new_covs
    "    
    "    def plot_creator(self, data):
    "        labels = self.predict(data)
    "        
    "        # create figure
    "        fig = Figure(figsize=(10, 8))
    "        canvas = FigureCanvas(fig)
    "        ax = fig.add_subplot(111)
    "        for i in range(self.n_mixtures):
    "            ax.scatter(data[:,0][labels == i], data[:, 1][labels == i], s=12, label='GMM-'+str(i))
    "        ax.set_xlabel('comp-1')
    "        ax.set_ylabel('comp-2')
    "        ax.set_title('Progress of EM Algorithm')
    "        ax.legend()
    "        fig.dpi = 200
    "        canvas.draw()
    "        image = np.asarray(canvas.buffer_rgba())
    "        return image
    "        
    "        
    "    def fit(self, data):
    "        n_features = data.shape[1]
    "        # intialize empty lists to store images and 
    "        # progress of EM algorithm per step
    "        progress_images = []
    "        self.log_likelihood_plot_list = []
    "        
    "        # initialize means
    "        kmeans_model = KMeans(n_clusters=self.n_mixtures)
    "        kmeans_model.fit(data)
    "        self.means = kmeans_model.centers
    "        
    "        # initialize cov
    "        self.covs = np.zeros((self.n_mixtures, n_features, n_features))
    "        data_labels = kmeans_model.predict(data)
    "        for i in range(self.n_mixtures):
    "            self.covs[i] = np.cov(data[data_labels == i].T)
    "        
    "        # EM - algorithm
    "        for step in range(self.max_iter):
    "            # for each data point find its responsibility
    "            # towards each gaussian
    "            resp = self.e_step(data)
    "            
    "            # re-estimation of model parameters
    "            alphas, means, covs = self.m_step(data, resp)
    "            
    "            # display progress after every 5 iterations
    "            if step % self.p_iter == 0 and self.plot_progress:
    "                progress_images.append(self.plot_creator(data))
    "            
    "            # break if convergence
    "            if np.allclose(self.alphas, alphas, rtol=1e-3, atol=1e-3) and \\
    "               np.allclose(self.means, means, rtol=1e-3, atol=1e-3) and \\
    "               np.allclose(self.covs, covs, rtol=1e-3, atol=1e-3):
    "               print('Converged after {}-th step'.format(step))
    "               break
    "                
    "            self.alphas = alphas
    "            self.means = means
    "            self.covs = covs
    "        
    "        self.log_likelihood_plot_list = self.log_likelihood_plot_list[1:]
    "        
    "        if self.plot_progress: 
    "            imageio.mimsave(self.gif_name, progress_images)
      "music_frames: (119920, 32)
      "speech_frames: (119920, 32)\n"
    "# Put your own paths for train/music and train/speech dirs
    "
    "music_train_path = './speech_music_classification/train/music/'
    "speech_train_path  = './speech_music_classification/train/speech/'
    "
    "music_frames = get_spec_frames(music_train_path)
    "print('music_frames:', music_frames.shape)
    "speech_frames = get_spec_frames(speech_train_path)
    "print('speech_frames:',speech_frames.shape)
    "\n"
    "# Put the number of mixture components and covrariance matrix type: {'diag','full'} here. 
    "
    "num_components = 2
      "Converged after 9-th step\n"
    "
    "music_model = GMM(n_mixtures=num_components, covar_type=covar_type) 
    "music_model.fit(music_frames)
    "\n"
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYkAAAEWCAYAAACT7WsrAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/Z1A+gAAAACXBIWXMAAAsTAAALEwEAmpwYAAAjh0lEQVR4nO3de5SddX3v8fdnLpkkkxskE8gNEhAQTGKKSDm18Uaslap4OVWppdVjxeWCFmzV1nNaD3q6bKWgbU9berhqlxVv2K4aaaRWa62t1EDj3km4CCTA3gnJBJidezIz+3v+eJ492RlnZybZe+bZs+fzWmuv2c/zey7fPSt5PrN/v+eiiMDMzGwkbVkXYGZmzcshYWZmNTkkzMysJoeEmZnV5JAwM7OaHBJmZlaTQ8JsDCS9WlKhanqLpFen72+U9IVT2ObQepLOkrRfUns6/S+SfqMx1Z+whvdI+rfx3o9NXg4JawmStktaN1H7i4iXRMS/NHB7T0fErIgYbNQ2zRrBIWFmZjU5JKxlSeqS9KeSdqSvP5XUVdX+UUk707bfkBSSXjTGbY/4zUVSp6R7JN0raZqkxen7XknbJP1Wje0tT/ffUTX7bEk/kLRP0v2SFlQt/+a0y6sv7Zq6sKrtwnReX7rMm6va5kv6B0l7Jf0ncO5YPq9NXQ4Ja2X/C7gMWAO8FLgU+H0ASb8I/DawDngR8Op6dyZpBvD3wBHgHcAA8A3gx8AS4HLgBkmvH+MmfwV4L7AQmAZ8ON3P+cA9wA1AD3Af8I00lDrTfd6frvebwN9KuiDd5l8Ch4FFwP9IX2Y1tVxISLpL0m5Jm8e4/DskbU3/4vrieNdnE+rdwCcjYndE9AKfAK5O294B3B0RWyLiIHBjnfuaA2wAngDem44tvBzoiYhPRsTRiHgSuB141xi3eXdEPBYRh4CvkIQdwDuBb0bEP0VEP3AzMAP4OZJQnAX8cbrP7wDrgavSQfG3Ax+PiAMRsRn4fJ2f21pcx+iLTDqfA/4C+JvRFpR0HvAx4BUR8YKkheNcm02sxcBTVdNPpfMqbRur2p6pvJF0FrC1Mh0Rs8awr8uATuCqOHbXzLOBxZL6qpZrB74/xvqfrXp/kOTgX6l96HNFRFnSMyTfVgaAZyKiXLXuU2lbD8n/+WeGtZnV1HLfJCLiX4Hnq+dJOlfSBkkPSvq+pBenTe8H/jIiXkjX3T3B5dr42kFyoK44K50HsBNYWtW2rPKm6kyjWWMMCEi6d/4I+GdJZ6TzngG2RcS8qtfsiLjilD7NMcd9LklK6y+mbcskVf/fPitt6yUJkWXD2sxqarmQqOE24Dcj4mUk/bp/lc4/Hzg/HRz8YdpPbZNXp6TplRdJv/3vS+pJB30/DlSuZ/gK8N50kHcm8Af17jwibgK+SBIUC4D/BPZJ+l1JMyS1S1op6eV17uorwC9Jujwdg/gdknGQfwceIPnW8dF0EP3VwJuAL6VdYF8HbpQ0U9JFwK/XWYu1uJYPCUmzSPpqvyppE/D/SAbtIPnqfR7JoOVVwO2S5k18ldYg9wGHql7TSbqUckAeeAj4Q4CI+Efgz4HvAo8DP0y3caSeAiLi/5AMXn8bmAu8kWQsYRuwB7gjnV/PPh4FfhX4v+k23wS8KR2DOJpOvyFt+yvg1yLikXT160i6rZ4l6Zq9u55arPWpFR86JGk5sD4iVkqaAzwaEYtGWO6vgQci4u50+p+B34uIH01owZa59BTSzUBXRAxkXY9Zs2j5bxIRsRfYJumXIem/lfTStPnvSU99TLsHzgeezKBMy4Ckt6bXUpwGfBr4hgPC7HgtFxKS7gH+A7hAUkHS+0hOhXyfpB8DW4Ar08W/BTwnaStJt8NHIuK5LOq2THwA2E1y2uog8MFsyzFrPi3Z3WRmZo3Rct8kzMyscVrqYroFCxbE8uXLsy7DzGxSefDBB/dERM9IbS0VEsuXL2fjxo2jL2hmZkMk1bzy3t1NZmZWk0PCzMxqckiYmVlNDgkzM6vJIWFmZjU5JMzMrCaHhJmZ1dRS10mYWfbK5aAcwWAEETCYTpfjp9vKEQyWhy+XLlu1TOXuQcfNAyKdjnS/kS5DZV6t5UeYVw4Iji1fvb/q5alaLln/2HTaDEPbTref/m6OrRND08k6tbfHsM9Zaa/eHhGcf+Zs3rh6MY3mkDBroIhgoBwcGShzpH+QIwNljg6UOTJQpn+wzGA5GCiXGRiM9P0YpgfLDJSrp4PB8rF5/Sc5PTCY7KOyvVoH8XI5+TyDw9vKVQf4YW1l3wouM29cvcghYTaagcEyh6sO0McO0ul0f5mjg4Mc6S+n7YNDB/GhA/tg+YTtR4fajw+ByvRE3TOzs120t4mOtjY62kVH2/HTyXvR3tZWtWzys7uzg/Y20a5kui39KTE0nbxIfrYl7yttErQPzT/Wpsr7YW3DtyEl+67VNjRfAMn+2iQEtLWB0nmqzFNl+lhbZV5lG20NWB6gLVkAcfz6qszk+HmVbaiqjaF9J+0jbY9h69TcXuXNOHFIWNM4OlBm3+F+9h0eSF/97E1/Vs/bd3iAfUeSn8e393O4v1x3HV0dbUzraKOro52ujja6OtuY1t5GV2cyPaurg/ndtdtHXr+dznalB+82OtID9nHT7ccO6sfakoN+e5uOC4X2tvE9MJhVOCSsISbqAD+js53Z0zvSVydzpnewdN6MoXmzujqZMe34A3RXR3t6ED82f1p6MK8c2CvT09rbxv0vM7PJxCFho9p/ZIDtew6wrer11HMH6Dt4LAiODIx+gJ85rX3o4D57egdzZ3Sy9LQZzKnM6+o4rr3yc076c9b0DjrbfUKe2URySBgARwYGefq5gzy558BQIFTe79535Lhll8ybwdnzZ3LR4hlDf82PdHA/7gDf1UGHD/Bmk45DYgoZLAfFFw7x5J79bEsD4Mk0EHb0HTruzJQFs6axYkE3rzq/hxU93ZyzoJvlC7pZPr+b6Z3t2X0IM5tQDokWExHs3neEJ3uTg//25w6k7/fz9PMH6R88lgSzuzpY0dPNy84+jbdfvJRzerpZkYbBnOmdGX4KM2sWDolJ6oUDR9n23AG2pWFQeb/9uQMcPDo4tNy0jjZWzO/mvIWzed1FZ3LOgm5WpGEwv3uaB2nN7IQcEk3uid79PLxzbxIGzx0bOO472D+0THubWHbaDFYs6Oayc+azYsFMViyYxYqebhbNmT50XreZ2clySDSxbXsOcPkt3xuaXjR3OisWdPNLqxaxYkH30GvZ6TN91o+ZjQuHRBN76KkXALj7vS/nshXzmTHNA8ZmNrEcEk0sXyzRPa2dV57X4ytszSwT7qNoYrlCHy9ZMtcBYWaZcUg0qYHBMlt27GX1krlZl2JmU5hDokn9ZPd+jgyUWbXUIWFm2XFINKl8oQTA6qXzsi3EzKY0h0STyhX7mD29g7NPn5l1KWY2hTkkmlS+UGLVkrm+EM7MMuWQaEJHB8o8vHOfxyPMLHN1hYSkGyUVJW1KX1fUWO56SZslbZF0Q9X8L1etu13SpnT+6yQ9KCmf/nxtPXVONo/t2sfRwTKrl8zLuhQzm+IacTHdZyPi5lqNklYC7wcuBY4CGyStj4jHI+KdVcvdApTSyT3AmyJiR7r+t4AlDah1UsgNDVr7m4SZZWsiupsuBB6IiIMRMQB8D3hb9QJKbkX6DuAegIj4r4jYkTZvAWZI6pqAWptCvtjHvJnJU9vMzLLUiJC4TlJO0l2SThuhfTOwVtJ8STOBK4Blw5ZZC+yKiJ+MsP7bgYci4sgIbUi6RtJGSRt7e3vr+RxNI5cOWvs23maWtVFDQtK30/GE4a8rgVuBc4E1wE7gluHrR8TDwKeB+4ENwCZgcNhiV5F+ixi275ek636gVn0RcVtEXBIRl/T09Iz2cZre4f5BHn12n7uazKwpjDomERHrxrIhSbcD62ts407gznS5TwGFqvU6SLqfXjZse0uBvwN+LSKeGEsNreCRZ/cxUA5WedDazJpAvWc3LaqafCtJ19JIyy1Mf55FEghfrGpeBzwSEdXBMQ/4JvB7EfGDemqcbPKFPsCD1mbWHOodk7gpPU01B7wG+BCApMWS7qta7l5JW4FvANdGRF9V27v46a6m64AXAR+vOkV2YZ21Tgq5QokFs6axaO70rEsxM6vvFNiIuLrG/B0kA9SV6bUn2MZ7Rpj3h8Af1lPbZJUvetDazJqHr7huIoeODvLYrn2s8k39zKxJOCSayNadJcqBnyFhZk3DIdFEKlda+55NZtYsHBJNJF8occacLs6Y40FrM2sODokmkiuWfH2EmTUVh0ST2H9kgCd69/v6CDNrKg6JJrGlWCLC4xFm1lwcEk0iX0wHrX1mk5k1EYdEk8gVSiyeO50Fs6bMHdHNbBJwSDSJfLHkriYzazoOiSZQOtTPtj0HWO0rrc2syTgkmsAWj0eYWZNySDSBnEPCzJqUQ6IJ5Asllp0+g9O6p2VdipnZcRwSTSBX7GO1r7Q2sybkkMjYCweO8szzh3xmk5k1JYdExioX0fn24GbWjBwSGauExEscEmbWhBwSGcsV+lixoJu5MzqzLsXM7Kc4JDKWL5R86quZNS2HRIZ69x1hR+mwbw9uZk3LIZGhzb6IzsyanEMiQ7lCCcmD1mbWvBwSGcoX+zi3ZxazujqyLsXMbEQOiQzlCiVfH2FmTc0hkZFdew+ze98RX2ltZk3NIZGRXCG90tohYWZNzCGRkXyhjzbBRYscEmbWvBwSGckVS5x/xmxmTGvPuhQzs5ocEhmICF9pbWaTgkMiAztKh3nuwFGPR5hZ06srJCTdKKkoaVP6uqLGctdL2ixpi6QbquZ/uWrd7ZI2DVvvLEn7JX24njqbTb7QB8CqpfMyrcPMbDSNuIrrsxFxc61GSSuB9wOXAkeBDZLWR8TjEfHOquVuAUrDVv8M8I8NqLGp5AolOtrEi8+cnXUpZmYnNBHdTRcCD0TEwYgYAL4HvK16AUkC3gHcUzXvLcA2YMsE1Dih8sUSF5w5m+mdHrQ2s+bWiJC4TlJO0l2SThuhfTOwVtJ8STOBK4Blw5ZZC+yKiJ8ASJoF/C7widF2LukaSRslbezt7a3vk0yAiEiutPZ4hJlNAqOGhKRvp+MJw19XArcC5wJrgJ3ALcPXj4iHgU8D9wMbgE3A4LDFrqLqWwRwI0k31v7R6ouI2yLikoi4pKenZ7TFM1d44RClQ/2sWjIv61LMzEY16phERKwby4Yk3Q6sr7GNO4E70+U+BRSq1usg6X56WdUqPwv8d0k3AfOAsqTDEfEXY6mlmflKazObTOoauJa0KCJ2ppNvJelaGmm5hRGxW9JZJIFwWVXzOuCRiBgKjohYW7XujcD+VggIgFyxj2ntbZx/hgetzaz51Xt2002S1gABbAc+ACBpMXBHRFROib1X0nygH7g2IvqqtvEuju9qamn5QokLF81mWocvUTGz5ldXSETE1TXm7yAZoK5Mrx1pubTtPaPs48ZTLK/plMtBvljiyjWLsy7FzGxM/OfsBHrq+YPsOzzAag9am9kk4ZCYQLmhK609aG1mk4NDYgLlCyW6Oto4b+GsrEsxMxsTh8QEyhVLvGTxHDra/Ws3s8nBR6sJMlgOthRLrPZN/cxsEnFITJBte/Zz4OignyFhZpOKQ2KC+EprM5uMHBITJFcoMXNaO+f0eNDazCYPh8QEyRdLrFw8l/Y2ZV2KmdmYOSQmwMBgmS07Sr4+wswmHYfEBHi8dz+H+8sejzCzScchMQEqg9Y+s8nMJhuHxATIF0rM7upg+fzurEsxMzspDokJkCuWWLlkLm0etDazScYhMc6ODpR5eOdej0eY2aTkkBhnj+3ax9GBss9sMrNJySExzvLF9EprP0PCzCYhh8Q4yxVKzJ3RybLTZ2RdipnZSXNIjLN8sY/VS+ciedDazCYfh8Q4Otw/yKPP7vP1EWY2aTkkxtGjz+6jfzB8ZpOZTVoOiXGUSwetV/lBQ2Y2STkkxlG+0Mf87mksnjs961LMzE6JQ2Ic5QrJnV89aG1mk5VDYpwcOjrIT3bvZ7UHrc1sEnNIjJOtO/cyWA6PR5jZpOaQGCf5Qh/gZ1qb2eTmkBgnuWKJhbO7OGOOB63NbPJySIyTfKHkbxFmNuk5JMbBgSMDPN67n5UetDazSc4hMQ627NhLhMcjzGzyqyskJN0oqShpU/q6osZy10vaLGmLpBuq5n+5at3tkjZVta2W9B/pOnlJk6ZzP5cOWvubhJlNdh0N2MZnI+LmWo2SVgLvBy4FjgIbJK2PiMcj4p1Vy90ClNL3HcAXgKsj4seS5gP9Dah1QuSLJRbNnc7C2ZMm18zMRjQR3U0XAg9ExMGIGAC+B7ytegEllyS/A7gnnfULQC4ifgwQEc9FxOAE1NoQ+ULJd341s5bQiJC4TlJO0l2SThuhfTOwVtJ8STOBK4Blw5ZZC+yKiJ+k0+cDIelbkh6S9NFaO5d0jaSNkjb29vY24OPUZ+/hfp7cc8DjEWbWEkYNCUnfTscThr+uBG4FzgXWADuBW4avHxEPA58G7gc2AJuA4d8KruLYtwhIusF+Hnh3+vOtki4fqb6IuC0iLomIS3p6ekb7OONus+/8amYtZNQxiYhYN5YNSbodWF9jG3cCd6bLfQooVK3XQdL99LKqVQrAv0bEnnSZ+4CLgX8eSy1ZyhfSkHB3k5m1gHrPblpUNflWkq6lkZZbmP48iyQQvljVvA54JCIKVfO+BaySNDMNkVcBW+updaLkiiWWnjaD07unZV2KmVnd6j276SZJa4AAtgMfAJC0GLgjIiqnxN5bdYbStRHRV7WNd3F8VxMR8YKkzwA/Srd9X0R8s85aJ4SvtDazVlJXSETE1TXm7yAZoK5Mrz3BNt5TY/4XSE6DnTT6Dh7l6ecPctWlZ2VdiplZQ/iK6wbKp4PW/iZhZq3CIdFAuXTQeuVih4SZtQaHRAPlCyWWz5/J3JmdWZdiZtYQDokGyhdLvj7CzFqKQ6JB9uw/QrHvkJ9pbWYtxSHRIPmhK60dEmbWOhwSDZIvlJDgJYvnZF2KmVnDOCQaJFcocc6CbmZP96C1mbUOh0SD5It9rPagtZm1GIdEA+zae5hde4/4pn5m1nIcEg1QufOrr7Q2s1bjkGiAXLFEm+AiD1qbWYtxSDRAvtDHeQtnM3NaIx4ZbmbWPBwSdYqI9EprdzWZWetxSNRpZ+kwe/Yf9XiEmbUkh0Sdcn5cqZm1MIdEnTYXS3S0iQsXedDazFqPQ6JOuWKJ88+YzfTO9qxLMTNrOIdEHSKCfKHP4xFm1rIcEnUovHCIFw72+8wmM2tZDok6DD3Tesm8bAsxMxsnDok65AolprW3cf6Zs7IuxcxsXDgk6pAv9vHiRbPp6vCgtZm1JofEKYoIcoWSr48ws5bmkDhFTz13kH2HB3xmk5m1NIfEKcpVnmntQWsza2EOiVOUL/TR1dHGeWd40NrMWpdD4hTlCiUuWjyHznb/Cs2sdfkIdwrK5WBzscRqD1qbWYtzSJyCJ/cc4MDRQVYtnZd1KWZm48ohcQryxT7Az7Q2s9ZXV0hIulFSUdKm9HVFjeWul7RZ0hZJN1TN/3LVutslbUrnd0r6vKS8pIclfayeOhstVygxo7Odc3s8aG1mra0RD2X+bETcXKtR0krg/cClwFFgg6T1EfF4RLyzarlbgFI6+ctAV0SskjQT2CrpnojY3oB665YvlFi5ZA7tbcq6FDOzcTUR3U0XAg9ExMGIGAC+B7ytegFJAt4B3JPOCqBbUgcwgyRc9k5AraMaGCyzZcdeXx9hZlNCI0LiOkk5SXdJOm2E9s3AWknz028FVwDLhi2zFtgVET9Jp78GHAB2Ak8DN0fE8yPtXNI1kjZK2tjb29uAj3NiT/Qe4FD/oMcjzGxKGDUkJH07HU8Y/roSuBU4F1hDckC/Zfj6EfEw8GngfmADsAkYHLbYVRz7FgFJ19QgsBhYAfyOpHNGqi8ibouISyLikp6entE+Tt1yhT4AP0PCzKaEUcckImLdWDYk6XZgfY1t3AncmS73KaBQtV4HSffTy6pW+RVgQ0T0A7sl/QC4BHhyLLWMp3yxxKyuDlbM7866FDOzcVfv2U2LqibfStK1NNJyC9OfZ5EEwhermtcBj0REoWre08Br03W6gcuAR+qptVFy6aB1mwetzWwKqHdM4qb0NNUc8BrgQwCSFku6r2q5eyVtBb4BXBsRfVVt7+L4riaAvwRmSdoC/Ai4OyJyddZat/7BMlt37mW1L6IzsymirlNgI+LqGvN3kAxQV6bXnmAb7xlh3n6S02CbymO79nF0oOxnSJjZlOErrk9CvpA+09qD1mY2RTgkTkKuWGLO9A7OOn1m1qWYmU0Ih8RJyBdKrF46j+TaPzOz1ueQGKMjA4M88uxeXx9hZlOKQ2KMHn12H/2D4WdImNmU4pAYo1w6aO1vEmY2lTgkxihfKHF69zSWzJuRdSlmZhPGITFGuWKJVUvmetDazKYUh8QYHO4f5LFd+3x9hJlNOQ6JMdi6cy+D5fCV1mY25TgkxiDvQWszm6IcEmOQK5RYMKuLM+dMz7oUM7MJ5ZAYg3yxj9VLPWhtZlOPQ2IUB44M8Pju/R6PMLMpySExiq0791IO3/nVzKYmh8Qohq609jcJM5uCHBKjyBf6OHPOdBZ60NrMpiCHxChyxZJPfTWzKcshcQL7DvfzZO8B3/nVzKYsh8QJbC7uBXwRnZlNXQ6JE8gX+wAPWpvZ1OWQOIFcocSSeTOYP6sr61LMzDLhkDiBfLHk6yPMbEpzSNRQOtjPU88d9HiEmU1pDoka8sXkIrrVS+ZlW4iZWYYcEjXkPGhtZuaQqCVfKHH2/JnMndmZdSlmZplxSNSQK5T8LcLMpjyHxAie23+EYt8hn9lkZlOeQ2IElUHrVR60NrMpziExgsozrVcumZNxJWZm2ao7JCTdKKkoaVP6uqLGctdL2ixpi6QbquavkfTDdN2Nki5N50vSn0t6XFJO0sX11jpWuWKJc3q6mT3dg9ZmNrU16pvEZyNiTfq6b3ijpJXA+4FLgZcCb5T0orT5JuATEbEG+Hg6DfAG4Lz0dQ1wa4NqHVW+UPKdX83MmLjupguBByLiYEQMAN8D3pa2BVDp15kL7EjfXwn8TSR+CMyTtGi8C9299zDP7j3MqqXzxntXZmZNr1EhcV3aJXSXpNNGaN8MrJU0X9JM4ApgWdp2A/Ankp4BbgY+ls5fAjxTtY1COu84kq5Ju6k29vb21v1Bhq609plNZmZjCwlJ307HE4a/riTpBjoXWAPsBG4Zvn5EPAx8Grgf2ABsAgbT5g8CH4qIZcCHgDtP5gNExG0RcUlEXNLT03Myq44oVyjRJrhokQetzcw6xrJQRKwby3KSbgfW19jGnaQBIOlTJN8MAH4duD59/1XgjvR9kWPfNgCWpvPGVb5Y4kULZ9HdNaZfjZlZS2vE2U3V4wRvJelaGmm5henPs0jGI76YNu0AXpW+fy3wk/T9PwC/lp7ldBlQioid9dZ7IhGRXmk9bzx3Y2Y2aTTiz+WbJK0hGYDeDnwAQNJi4I6IqJwSe6+k+UA/cG1E9KXz3w/8maQO4DDJmUwA95GMXTwOHATe24BaT2jX3iPs2X/E4xFmZqm6QyIirq4xfwfJQb4yvbbGcv8GvGyE+QFcW299JyNX6AP8TGszswpfcV0lXyzR3iYPWpuZpRwSVXKFEuefMZvpne1Zl2Jm1hQcEqmISJ5p7SutzcyGOCRSxb5DPH/gqMcjzMyqOCRSlTu/+swmM7NjHBKpXLFEZ7u44MzZWZdiZtY0HBKpfKHEi8+cQ1eHB63NzCocElSutO7zeISZ2TAOCeDp5w+y9/CAz2wyMxvGIQH0D5Z5w8ozufjske5ybmY2dflWp8CLFs7m1l/9qTuDmJlNef4mYWZmNTkkzMysJoeEmZnV5JAwM7OaHBJmZlaTQ8LMzGpySJiZWU0OCTMzq0nJo6Rbg6Re4Kk6NrEA2NOgchrJdZ0c13VyXNfJacW6zo6InpEaWiok6iVpY0RcknUdw7muk+O6To7rOjlTrS53N5mZWU0OCTMzq8khcbzbsi6gBtd1clzXyXFdJ2dK1eUxCTMzq8nfJMzMrCaHhJmZ1eSQACT9oqRHJT0u6feyrqdC0l2SdkvanHUtFZKWSfqupK2Stki6PuuaACRNl/Sfkn6c1vWJrGuqJqld0n9JWp91LRWStkvKS9okaWPW9VRImifpa5IekfSwpP/WBDVdkP6eKq+9km7Iui4ASR9K/81vlnSPpOkN3f5UH5OQ1A48BrwOKAA/Aq6KiK2ZFgZIeiWwH/ibiFiZdT0AkhYBiyLiIUmzgQeBt2T9+5IkoDsi9kvqBP4NuD4ifphlXRWSfhu4BJgTEW/Muh5IQgK4JCKa6sIwSZ8Hvh8Rd0iaBsyMiL6MyxqSHjOKwM9GRD0X7zailiUk/9YviohDkr4C3BcRn2vUPvxNAi4FHo+IJyPiKPAl4MqMawIgIv4VeD7rOqpFxM6IeCh9vw94GFiSbVUQif3pZGf6aoq/gCQtBX4JuCPrWpqdpLnAK4E7ASLiaDMFROpy4ImsA6JKBzBDUgcwE9jRyI07JJID3DNV0wWa4KA3GUhaDvwM8EDGpQBDXTqbgN3AP0VEU9QF/CnwUaCccR3DBXC/pAclXZN1MakVQC9wd9o9d4ek7qyLGuZdwD1ZFwEQEUXgZuBpYCdQioj7G7kPh4SdEkmzgHuBGyJib9b1AETEYESsAZYCl0rKvItO0huB3RHxYNa1jODnI+Ji4A3AtWn3ZtY6gIuBWyPiZ4ADQDONE04D3gx8NetaACSdRtLzsQJYDHRL+tVG7sMhkfQtLquaXprOsxrSPv97gb+NiK9nXc9waffEd4FfzLgUgFcAb077/78EvFbSF7ItKZH+FUpE7Ab+jqTrNWsFoFD1LfBrJKHRLN4APBQRu7IuJLUO2BYRvRHRD3wd+LlG7sAhkQxUnydpRfpXwruAf8i4pqaVDhDfCTwcEZ/Jup4KST2S5qXvZ5CciPBIpkUBEfGxiFgaEctJ/m19JyIa+pfeqZDUnZ54QNqd8wtA5mfRRcSzwDOSLkhnXQ5kfhJJlatokq6m1NPAZZJmpv83LycZJ2yYjkZubDKKiAFJ1wHfAtqBuyJiS8ZlASDpHuDVwAJJBeB/R8Sd2VbFK4CrgXza/w/wPyPivuxKAmAR8Pn0zJM24CsR0TSnmzahM4C/S44rdABfjIgN2ZY05DeBv03/aHsSeG/G9QBDYfo64ANZ11IREQ9I+hrwEDAA/BcNvj3HlD8F1szManN3k5mZ1eSQMDOzmhwSZmZWk0PCzMxqckiYmVlNDgmzlKQ/kvQaSW+R9LF03iclrUvf3yBpZgP39xZJF1VND+3LrFn4FFizlKTvkNyI71PA1yLiB8Pat3OSd02V1B4RgzXaPgesj4ivnXLRZuPMIWFTnqQ/AV5Pcv+bJ4BzgW0kt4Q4B1hPcl+cm4FHgT0R8RpJvwB8AuhK13tveqvy7cCXSS68ugmYDVwDTAMeJ7kYcU263VL6ejvwB6ShIenydH8dJHcF+GBEHEm3/XngTSR3uv3liHhE0quAP0s/UgCvTO/Sa1YXdzfZlBcRHwHeB3wOeDmQi4jVEfHJqmX+nOQWzK9JA2IB8PvAuvQmeRuB367a7HMRcXFEfAn4ekS8PCJeSnLLhPdFxL+T3P7lIxGxJiKeqKyYPjTmc8A7I2IVSVB8sGrbe9J93gp8OJ33YeDa9AaHa4FDjfjdmDkkzBIXAz8GXszY7n1zGXAR8IP09iS/Dpxd1f7lqvcrJX1fUh54N/CSUbZ9AclN2x5Lpz9P8oyFispNFR8ElqfvfwB8RtJvAfMiYmAMn8FsVFP+3k02tUlaQ/JX+1JgD8lDW5Qe+E/02EyRPLPiqhrtB6ref47k6X0/lvQekvtx1eNI+nOQ9P9wRPyxpG8CV5AE1+sjIvMbHNrk528SNqVFxKa0i+Yxkm8G3wFen3YBDe+y2UcyvgDwQ+AVkl4EQ3dVPb/GbmYDO9NbrL+7xvaqPQosr2ybZAzjeyf6HJLOjYh8RHyaZAzjxSda3mysHBI25UnqAV6IiDLw4hM8r/s2YIOk70ZEL/Ae4B5JOeA/qH1g/gOSp/f9gONvX/4l4CPpE9jOrcyMiMMkdz79atpFVQb+epSPcYOkzWkt/cA/jrK82Zj47CYzM6vJ3yTMzKwmh4SZmdXkkDAzs5ocEmZmVpNDwszManJImJlZTQ4JMzOr6f8DgQApYjRHdD8AAAAASUVORK5CYII=
    "# Plot the log likelihood for music model
    "plt.plot(music_model.log_likelihood_plot_list)
    "plt.xlabel('#iterations')
      "Converged after 8-th step\n"
    "speech_model = GMM(n_mixtures=num_components, covar_type=covar_type)
    "speech_model.fit(speech_frames)\n"
      "image/png": "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
    "# Plot the log likelihood for speech model
    "plt.plot(speech_model.log_likelihood_plot_list)
    "plt.xlabel('#iterations')
    "## Testing
      "First few test files:
      "
      " ['./speech_music_classification/test/speech_2.wav', './speech_music_classification/test/speech_15.wav', './speech_music_classification/test/music_12.wav', './speech_music_classification/test/speech_7.wav', './speech_music_classification/test/music_2.wav', './speech_music_classification/test/speech_19.wav', './speech_music_classification/test/speech_8.wav', './speech_music_classification/test/speech_5.wav', './speech_music_classification/test/music_23.wav', './speech_music_classification/test/speech_20.wav'] 
      "
      "
      "First few test labels:
      "
      " [0, 0, 1, 0, 1, 0, 0, 0, 1, 0]\n"
    "test_path = './speech_music_classification/test/'
    "label_dict = {'speech':0, 'music':1}
    "
    "test_labels = []
    "test_dir_list = glob.glob(test_path + '*.wav')
    "for i in range(len(test_dir_list)):
    "    f = test_dir_list[i]
    "    label_str = f.split('/')[-1].split('_')[0]
    "    test_labels.append(label_dict[label_str])
    "
    "print('First few test files:\\n\\n', test_dir_list[0:40:4],'\\n\\n')
    "print('First few test labels:\\n\\n', test_labels[0:40:4])\n"
    "# predict
    "
    "pred_labels = []
    "for i in range(len(test_dir_list)):
    "
    "    y,sr = librosa.load(test_dir_list[i], sr=None)
    "    y_spec = spectrogram(y)
    "    #print(y_spec.shape)
    "
    "    ll_speech = speech_model.get_log_likelihood(y_spec)
    "    ll_music = music_model.get_log_likelihood(y_spec)
    "    pred_label = 0 if ll_speech > ll_music else 1
    "    pred_labels.append(pred_label)
    "
    "test_labels = np.array(test_labels)
      "Number of correctly classified examples: 43, out of 48 test samples
      "error rate:  10.416666666666671\n"
    "# Calculate error rate
    "
    "correct = (test_labels == pred_labels).sum()
    "print('Number of correctly classified examples: {}, out of {} test samples'.format(correct, len(test_dir_list)))
    "accuracy = 100. * correct / len(test_dir_list)
    "### Observations:
    "#### a) As number of mixture components increases, the mixture of Gaussians better approximates the data distribution. So we see decrease in error rate. 
    "### Note:
    "#### Here the classification is done based on log-likelihood of the spectrogram assuming its feature frames are independently drawn. Another criteria would be to use majority count over predicted labels of the feature vectors. 
