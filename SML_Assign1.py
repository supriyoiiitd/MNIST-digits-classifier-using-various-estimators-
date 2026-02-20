import kagglehub
import idx2numpy as idn
import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("hojjatk/mnist-dataset")

print("Path to dataset files:", path)
def image28x28to784(matrix):
    if matrix.shape != (28, 28):
        print("Input matrix must be 28x28")
        return None
    flat_list=[]
    for i in range(28):
        for j in range(28):
            flat_list.append(matrix[j][i])
    
    for i in range(len(flat_list)):
        flat_list[i]/=255.0
    return flat_list
def manual_log_gaussian(x,mu,sigma):
    d =len(x)
    sigma_inv= np.linalg.inv(sigma)
    _, logdet= np.linalg.slogdet(sigma)
    diff= x-mu
    exponent= -0.5*np.dot(diff.T,np.dot(sigma_inv,diff))
    normalization= -0.5*(d*np.log(2*np.pi)+logdet)
    return exponent + normalization


training_img_path= os.path.join(path,"train-images.idx3-ubyte")
raw_trainset= idn.convert_from_file(training_img_path)
training_label_path=os.path.join(path,"train-labels.idx1-ubyte")
raw_traininglabels= idn.convert_from_file(training_label_path)

train_X = []
train_y = []
for digit in [0, 1, 2]:
    all_indices = np.where(raw_traininglabels==digit)[0]
    selected_indices = np.random.choice(all_indices, 100, replace=False)
    for idx in selected_indices:
        matrix = raw_trainset[idx]
        vector = image28x28to784(matrix)
        train_X.append(vector)
        train_y.append(digit)

testing_img_path= os.path.join(path,"t10k-images.idx3-ubyte")
raw_testset= idn.convert_from_file(testing_img_path)
testing_label_path=os.path.join(path,"t10k-labels.idx1-ubyte")
raw_testinglabels= idn.convert_from_file(testing_label_path)

test_x=[]
test_y=[]

for digit in [0, 1, 2]:
    all_indices = np.where(raw_testinglabels==digit)[0]
    selected_indices= np.random.choice(all_indices, 100, replace=False)
    for idx in selected_indices:
        matrix= raw_testset[idx]
        vector= image28x28to784(matrix)
        test_x.append(vector)
        test_y.append(digit)

train_X = np.array(train_X)
train_y = np.array(train_y)
#MLE estimation
means= {}
covariances= {}
classes= [0, 1, 2]

for c in classes:
    X_c=train_X[train_y==c]
    N= X_c.shape[0]
    means[c]= np.mean(X_c, axis=0)
    diff= X_c-means[c]

    covariances[c]= np.dot(diff.T, diff)/ N 
    covariances[c]+= np.eye(784)*1e-6 #singular matrix fix
# LDA AND QDA

shared_cov = np.mean(list(covariances.values()), axis=0)# FOR LDA

def classify_dataset(X_test, is_lda=False):
    predictions = []
    for x in X_test:
        log_likelihoods = []
        for c in classes:
            mu= means[c]
            sigma= shared_cov if is_lda else covariances[c]
            val= manual_log_gaussian(x, mu, sigma)
            log_likelihoods.append(val)
        predictions.append(np.argmax(log_likelihoods))
    return np.array(predictions)

lda_preds= classify_dataset(test_x, is_lda=True)
qda_preds= classify_dataset(test_x, is_lda=False)

#accuracy report
def calculate_accuracy(y_true, y_pred):
    indicator_sum= np.sum(y_true == y_pred) 
    return indicator_sum/ len(y_true) 

lda_accuracy= calculate_accuracy(test_y, lda_preds)
qda_accuracy= calculate_accuracy(test_y, qda_preds)

print(f"LDA Accuracy: {lda_accuracy * 100:.2f}%")
print(f"QDA Accuracy: {qda_accuracy * 100:.2f}%")

def plot_tsne(data, labels, title):

    tsne= TSNE(n_components=2, random_state=42)
    data_2d= tsne.fit_transform(data)
    
    plt.figure(figsize=(8, 6))
    colors= ['red', 'green', 'blue']
    for i, c in enumerate(classes):
        mask= labels == c
        plt.scatter(data_2d[mask, 0], data_2d[mask, 1], c=colors[i], label=f'Digit {c}', alpha=0.6)
    
    plt.title(title)
    plt.legend()
    plt.show()

plot_tsne(train_X, train_y, "t-SNE Plot: MNIST Train Set (Digits 0, 1, 2)")
plot_tsne(np.array(test_x), np.array(test_y), "t-SNE Plot: MNIST Test Set (Digits 0, 1, 2)")
sample_idx = 0
sample_x = test_x[sample_idx]
true_label = test_y[sample_idx]

print(f"\n Discriminant Values for Test Sample {sample_idx} (True Label: {true_label})")
for c in classes:
    disc_val = manual_log_gaussian(sample_x, means[c], covariances[c])
    print(f"Class {c} Discriminant Value (Log-Likelihood): {disc_val:.4f}")
# Generate Test Set Plot
plot_tsne(np.array(test_x), np.array(test_y), "t-SNE Plot: MNIST Test Set (Digits 0, 1, 2)")