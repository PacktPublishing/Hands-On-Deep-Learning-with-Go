# Pengenalan AI

Terori Menjelaskan :
* Definisi Kecerdasan Buatan
* Training
* Kalkulus Predikat dan Preposisi
* Himpunan
* Persaman Garis Linear
* Pemodelan Persamaan
* Jaringan Saraf Tiruan
* Layer, Label, Supervised dan Unsupervised
* Machine Learning, Deep Learning

## golang

Pengenalan Golang : https://youtu.be/R0EF4apbCag

* fork terlebih dahulu, clone ke pc
* buat folder NPM di dalam chapter01/Kelas/NPM
* ketik : go mod init github.com/bukped/ai/Chapter01/KELAS/NPM
* buat file main.go dan ketikkan perintah hello word
* go mod tidy
* go run .

## gorgonia.org

Pengenalan Gorgonia : https://youtu.be/ckLI1xsyF0c

Pada bagian ini dikenalkan gorgonia.org. Gorgonia memerlukan 3 langkah pengerjaan:
1. Membuat graph komputasi
   ![image](https://user-images.githubusercontent.com/11188109/221063539-122804b2-96a8-49b4-adf8-d6c28e53fcf5.png)
2. Data Input
3. Eksekusi graph komputasi

Graph adalah fungsi yang mengelola semua variabel. Variabel di deep learning dikenal dengan Tensor. Jenis jenis tensor terlihat pada gambar berikut.

![image](https://user-images.githubusercontent.com/11188109/221063068-f3b97fe7-8482-4001-b072-5abf494ea7e9.png)

hello word : https://gorgonia.org/tutorials/hello-world/

### Membuat kode program

Membuat persamaan : https://youtu.be/CS1UKDK45t0

Berikut adalah contoh pengerjaan pemrograman fungsi dengan menggunakan gorgonia.

#### Buat kode program fungsi : c = a + b

1. Deklarasi package dan import library gorgonia
   ```go
   package main
   import (
        "fmt"
        "log"
        . "gorgonia.org/gorgonia"
    )
   ```
2. Deklarasikan fungsi main, dan inisiasi NewGraph() untuk deklarasi membuat graph komputasi
   ```go
   func main() {
     g := NewGraph()
   }
   ```
3. Deklarasikan tensor yang akan terlibat, disini a dan b sebagai inputan dari graph komputasi.
   ```go
   a = NewScalar(g, Float64, WithName("a"))
   b = NewScalar(g, Float64, WithName("b"))
   ```
4. Definisikan fungsi c=a+b dalam graph komputasi gorgonia.
   ```go
   c, err = Add(a,b)
   ```
5. Buat VM object agar bisa menjalankan model fungsi g yang dideklarasikan pada langkah 2.
   ```go
   machine := NewTapeMachine(g)
   ```
6. Untuk menjalankan model maka gunakan method RunAll() dari variabel VM yang dibuat. Jangan lupa isi inisiasi inputan a dan b.
   ```go
   Let(a, 1.0)
   Let(b, 2.0)
   machine.RunAll()
   ```

#### Buat kode program fungsi z = Wx dimana W adalah matriks n kali n. x adalah vektor ukuran n. dengan n = 2.1957

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
2. Deklarasikan fungsi main, dan inisiasi NewGraph() untuk deklarasi membuat graph komputasi
   ```go
   func main() {
     g := NewGraph()
   }
   ```
3. Deklarasikan tensor yang akan terlibat, disini matriks W dan x sebagai inputan dari graph komputasi.
   ```go
   //deklarasi W, dengan bobot inisiasi matB
   matB := []float64{0.9,0.7,0.4,0.2}
   matT := tensor.New(tensor.WithBacking(matB), tensor.WithShape(2, 2))
   mat := G.NewMatrix(g,
           tensor.Float64,
           G.WithName("W"),
           G.WithShape(2, 2),
           G.WithValue(matT),
   )
   
   // deklarasi x dengan inisiasi bobot vecB
   vecB := []float64{5,7}

   vecT := tensor.New(tensor.WithBacking(vecB), tensor.WithShape(2))

   vec := G.NewVector(g,
           tensor.Float64,
           G.WithName("x"),
           G.WithShape(2),
           G.WithValue(vecT),
   )
   ```
4. Definisikan fungsi z=Wx dalam graph komputasi gorgonia. Karena perkalian maka menggunakan rumus multification.
   ```go
   z, err := G.Mul(mat, vec)
   ```
5. Buat VM object agar bisa menjalankan model fungsi g yang dideklarasikan pada langkah 2.
   ```go
   machine := G.NewTapeMachine(g)
   ```
6. Untuk menjalankan model maka gunakan method RunAll() dari variabel VM yang dibuat. Jangan lupa isi inisiasi inputan a dan b.
   ```go
   machine.RunAll()
   //melihat hasil output
   fmt.Println(z.Value().Data())
   ```
   
Untuk melakukan visualisasi graph komputasi, kita bisa menggunakan ioutil.
```go
ioutil.WriteFile("simple_graph.dot", []byte(g.ToDot()), 0644)
```

Konversi ke file SVG bisa dilakukan dengan menggunakan [dot](https://www.mankier.com/1/dot)
```sh
dot -Tsvg simple_graph.dot -O
```

#### Buat kode program fungsi z = Wx + b

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
2. Deklarasikan fungsi main, dan inisiasi NewGraph() untuk deklarasi membuat graph komputasi
   ```go
   func main() {
     g := NewGraph()
   }
   ```
3. Deklarasikan tensor yang akan terlibat, disini matriks W dan x sebagai inputan dari graph komputasi.
   ```go
   //deklarasi W, dengan bobot inisiasi matB
   matB := []float64{0.9,0.7,0.4,0.2}
   matT := tensor.New(tensor.WithBacking(matB), tensor.WithShape(2, 2))
   mat := G.NewMatrix(g,
           tensor.Float64,
           G.WithName("W"),
           G.WithShape(2, 2),
           G.WithValue(matT),
   )
   
   // deklarasi x dengan inisiasi bobot vecB
   vecB := []float64{5,7}

   vecT := tensor.New(tensor.WithBacking(vecB), tensor.WithShape(2))

   vec := G.NewVector(g,
           tensor.Float64,
           G.WithName("x"),
           G.WithShape(2),
           G.WithValue(vecT),
   )
   
   //tambah deklarasi b
   b := G.NewScalar(g,
        tensor.Float64,
        G.WithName("b"),
        G.WithValue(3.0)
   )
   ```
4. Definisikan fungsi z=Wx dalam graph komputasi gorgonia. Karena perkalian maka menggunakan rumus multification.
   ```go
   a, err := G.Mul(mat, vec)
   z, err := G.Add(a, b)
   ```
5. Buat VM object agar bisa menjalankan model fungsi g yang dideklarasikan pada langkah 2.
   ```go
   machine := G.NewTapeMachine(g)
   ```
6. Untuk menjalankan model maka gunakan method RunAll() dari variabel VM yang dibuat. Jangan lupa isi inisiasi inputan a dan b.
   ```go
   machine.RunAll()
   //melihat hasil output
   fmt.Println(z.Value().Data())
   ```

## Kerjakan
1. Buat kode program persamaan garis(selain persamaan linear, misal persamaan parabola, elips,sinus dll), yang berbeda satu sama lain dalam satu kelas. dengan gorgonia simpan di folder Chapter01/KELAS/NPM (nilai 60)
2. Pull request dengan menggunakan judul : 1-KELAS-NPM-NAMA
3. Pada deskripsi lampirkan skrinsut running program
4. Lampirkan juga sertifikat dari :
   * https://www.mygreatlearning.com/academy/learn-for-free/courses/go-programming-language (nilai 10)
   * https://www.mindluster.com/certificate/3394 (nilai 10)
   * https://www.codecademy.com/learn/learn-go (nilai 10)
   * https://www.coursera.org/specializations/google-golang (nilai 10)
