# Neural Network

Pada bagian ini akan dipelajari :
* Dasar Neural Network
* Fungsi Aktifasi
* Gradient Descent dan Backpropagation
* Advanced gradient descent algorithms

## Membangun Neural Network

Tujuannya adalah melakukan proses optimisasi yang disebut Stochastic Gradient Descent(SGD) atau backpropagation dari setiap bobot(w) di layer masing-masing. Seperti sebelumnya, kita buat dahulu model graph nya.

* Input data(l0) : 4 x 3 matrix
* Output data(pred) : 4 x 1 vektor
* inisialisasi bobot(w) dengan 3x1 vektor
* Sigmoid nonlinearity(SGD) melakukan optimasi pada w0
* dua layer : 
  * l0 adalah Input data
  * l1 output SGD dari l0 x w0
