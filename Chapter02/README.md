# Neural Network

Pada bagian ini akan dipelajari :
* Dasar Neural Network
* Fungsi Aktifasi
* Gradient Descent dan Backpropagation
* Advanced gradient descent algorithms

## Membangun Neural Network

Tujuannya adalah melakukan proses optimisasi yang disebut Stochastic Gradient Descent(SGD) atau backpropagation dari setiap bobot(w) di layer masing-masing. Seperti sebelumnya, kita buat dahulu model graph nya.

![image](https://user-images.githubusercontent.com/11188109/224580308-b8a84b6a-cf69-490a-8441-a33de7c2fa2c.png)

* Input data(L0) : 4 x 3 matrix
* Output data(pred) : 4 x 1 vektor
* inisialisasi bobot(w) dengan 3x1 vektor
* Sigmoid nonlinearity(SGD) melakukan optimasi pada w0
* Neural network terdiri dari dua layer : 
  * L0 : adalah Input data
  * L1 : output SGD dari l0 x w0

Kita mulai dengan membuat package dan melakukan import gorgonia
```go
package main

import (
    "fmt"
    "io/ioutil"
    "log"

    . "gorgonia.org/gorgonia"
    "gorgonia.org/tensor"
)
```

### Neural Network dalam type struct dan fungsi go

Dalam go, kita bisa mendefinisikan neural network dalam type struct.

```go
type nn struct {
    g *ExprGraph
    w0, w1 *Node

    pred *Node
}
```

Kemudian kita akan membuat fungsi baru newNN dengan inputan g yaitu graph komputasi gorgonia. Lakukan inisiasi bobot awal dan return berupa struct nn diatas.

```go
func newNN(g *ExprGraph) *nn {
    // Create node for w/weight (needs fixed values replaced with random values w/mean 0)
    wB := []float64{-0.167855599, 0.44064899, -0.99977125}
    wT := tensor.New(tensor.WithBacking(wB), tensor.WithShape(3, 1))
    w0 := NewMatrix(g,
        tensor.Float64,
        WithName("w"),
        WithShape(3, 1),
        WithValue(wT),
    )
    return nn{
        g: g,
        w0: w0,
    }
}
```

buat fungsi grouping setiap nodes, agar bisa dilakukan kalkulasi gradien dari setiap n-layers yang ada

```go
func (m *nn) learnables() Nodes {
    return Nodes{m.w0}
}
```

Fungsi utama neural network yang akan kita buat menjadi 

```go
func (m *nn) fwd(x *Node) (err error) {
    var l0, l1 *Node

    // Set first layer to be copy of input
    l0 = x

    // Dot product of l0 and w0, use as input for Sigmoid
    l0dot := Must(Mul(l0, m.w0))

    // Build hidden layer out of result
    l1 = Must(Sigmoid(l0dot))
    // fmt.Println("l1: \n", l1.Value())

    m.pred = l1
    return

}
```

#### Buat kode program utama

1. Deklarasi package dan import library gorgonia
   ```go
   package main
   import (
        "fmt"
        "log"

        G "gorgonia.org/gorgonia"
        "gorgonia.org/tensor"
   )
   ```
2. Deklarasikan fungsi main, dan inisiasi NewGraph() untuk deklarasi membuat graph komputasi, serta newNN(g) dari fungsi yang sudah dibuat sebelumnya
   ```go
   func main() {
    rand.Seed(31337)

    intercept Ctrl+C
    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
    doneChan := make(chan bool, 1)

    // Create graph and network
    g := NewGraph()
    m := newNN(g)
   }
   ```
3. Deklarasikan tensor yang akan terlibat, disini y dan x sebagai inputan dari graph komputasi.
   ```go
   // Set input x to network
   xB := []float64{0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1}
   xT := tensor.New(tensor.WithBacking(xB), tensor.WithShape(4, 3))
   x := NewMatrix(g,
        tensor.Float64,
        WithName("X"),
        WithShape(4, 3),
        WithValue(xT),
    )
   // Define validation dataset
    yB := []float64{0, 0, 1, 1}
    yT := tensor.New(tensor.WithBacking(yB), tensor.WithShape(4, 1))
    y := NewMatrix(g,
        tensor.Float64,
        WithName("y"),
        WithShape(4, 1),
        WithValue(yT),
    )
   
   ```
4. Definisikan fungsi fwd dalam graph komputasi gorgonia. Disini suda dapat prediksi pertama, kita akan melakukan optimisasi dengan mendefinisikan lossess dan cost.
   ```go
   m.fwd(x)
   
   //optimasi
   losses := Must(Sub(y, m.pred))
   cost := Must(Mean(losses))
   var costVal Value
   Read(cost, costVal)
   //komputasi gradient
   Grad(cost, m.learnables()...)
   ```
5. Buat VM object agar bisa menjalankan model fungsi g dan m yang dideklarasikan pada langkah sebelumnya. vanilla SGD digunakan sebagai solver.
   ```go
   // Instantiate VM and Solver
   vm := NewTapeMachine(g, BindDualValues(m.learnables()...))
   solver := NewVanillaSolver(WithLearnRate(0.001), WithClip(5))
   // solver := NewRMSPropSolver()
   ```
6. Untuk menjalankan model dengan iterasi sebanyak 10000 kali.iterasi looping inilah yang kita sebut sebagai training.
   ```go
   for i := 0; i < 10000; i++ {
        vm.RunAll()
        solver.Step(NodesToValueGrads(m.learnables()))
        fmt.Println("\nState at iter", i)
        fmt.Println("Cost: \n", cost.Value())
        fmt.Println("Weights: \n", m.w0.Value())
        // vm.Set(m.w0, wUpd)
        // vm.Reset()
    }
    fmt.Println("Output after Training: \n", m.pred.Value())
   ```

## Gradien Fungsi aktifasi

![image](https://user-images.githubusercontent.com/11188109/224583377-863ad7ba-8577-44d5-a4d3-4018a843cbe3.png)

Optimalisasi gradien bisa dengan melihat :
* Kenaikan : ascending
* Penurunan : descending

Fungsi aktifasi disebut juga sebagai transfer function. Berfungsi untuk memudahkan optimasi. Sebagai pertimbangan :
* Simple : fungsi dan persamaannya
* DIferensiabilitas : memiliki nilai yang berbeda
* Kontinuitas : non diskrit, garisnya menyambung tidak putus
* Monotonitas : Menggunakan satu fungsi saja agar menghemat komputasi

### Step Function

![image](https://user-images.githubusercontent.com/11188109/224581866-995ce64e-6762-483f-908f-77b46c373084.png)

Fungsi aktifasi paling sederhana, mengeluarkan angka 0 dan 1 saja.
```go
func step(x) {
    if x >= 0 {
        return 1
    } else {
        return 0
    }
}
```

### Linear Function

![image](https://user-images.githubusercontent.com/11188109/224582034-235ecf4c-dc3c-43ac-936c-2dc35c72db34.png)

Fungsi aktifasi yang sama dengan step, tetapi menggunakan persamaan garis liniear y = ax+b. Penggunaannya tidak akan banyak membantu untuk optimasi gradien. karena nilainya akan sama.
```go
func linear(x){
   a:=1
   b:=0
   return a * x + b
}
```

### Rectified Linear Units

![image](https://user-images.githubusercontent.com/11188109/224582329-a60e98cb-c099-418a-9196-4762c106a662.png)

Fungsi aktifasi yang paling populer karena sifatnya yang menghasilkan non-linear dan cepat. Menggabungkan step dan liniear function. Menghasilkan 0 jika nilai inputan negatif. Kekurangannya adalah, jika hasil 0 maka neuran atau node akan mati dan tidak akan di hidupkan lagi. 
```go
func relu(x){
   return Max(0,x)
}
```

### Leaky ReLU

![image](https://user-images.githubusercontent.com/11188109/224582825-e17b840c-632e-40cc-98f4-68a9df2e1096.png)

Untuk memperbaiki peluang ReLU yang menghasilkan nilai 0 sehingga membuat node mati. Maka diatasi dengan membuat nilai 0 memiliki nilai yang sangat-sangat kecil, sehingga tidak mematikan node neuron.
```go
func leaky_relu(x) {
    if x >= 0 {
        return x
    } else {
        return 0.01 * x
    }
}
```

### Sigmoid Function

![image](https://user-images.githubusercontent.com/11188109/224583111-8cc1a636-1387-4d7c-8b53-a27163f9aac7.png)

Memiliki hasil yang lebih baik daripada ReLU, hanya saja beban komputasinya tinggi. Fungsi sigmoid merupakan fungsi non linear yang berarti tidak memperlihatkan garis yagn lurus, tetapi berbeda di setiap titiknya.
```go
func sigmoid(x){
    return 1 / (1 + Exp(-x))
}
```

### Tanh

![image](https://user-images.githubusercontent.com/11188109/224583222-d86a1bb9-4db2-4d09-9d8b-9c0c825b8386.png)

Jika kita menginginkan fungsi sigmoid yang menghasilkan juga nilai negatif maka kita bisa menggunakan tanh.
```go
func tanh(x){
  return 2 * (1 + Exp(-2*x)) - 1
}
```
