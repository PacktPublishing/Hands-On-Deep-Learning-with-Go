# Neural Network

Pada bagian ini akan dipelajari :
* Dasar Neural Network
* Fungsi Aktifasi
* Gradient Descent dan Backpropagation
* Advanced gradient descent algorithms

## Membangun Neural Network

Tujuannya adalah melakukan proses optimisasi yang disebut Stochastic Gradient Descent(SGD) atau backpropagation dari setiap bobot(w) di layer masing-masing. Seperti sebelumnya, kita buat dahulu model graph nya.

* Input data(L0) : 4 x 3 matrix
* Output data(pred) : 4 x 1 vektor
* inisialisasi bobot(w) dengan 3x1 vektor
* Sigmoid nonlinearity(SGD) melakukan optimasi pada w0
* Neural network terdiri dari dua layer : 
  * L0 : adalah Input data
  * L1 : output SGD dari l0 x w0

### Neural Network dalam type struct go

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
