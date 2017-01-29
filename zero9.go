package main

import (
    "os"
    "fmt"
    "log"
    "math"
    "math/rand"
    "bufio"
    "strconv"
    "strings"
    "github.com/gonum/matrix/mat64"
)


type ITEM struct {
    x *mat64.Dense
    y int
}


func load_one(infn string) (dat []*ITEM, err error) {
    fin, err := os.Open(infn)
    if err != nil {
        log.Fatal(err)
    }
    defer fin.Close()
    println(infn)
    dat = make([]*ITEM, 0, 1000)
    scanner := bufio.NewScanner(fin)
    for scanner.Scan() {
        cols := strings.Fields(scanner.Text())
        if len(cols) != 785 { log.Fatalf("got column %d", len(cols)) }
        var n int64
        n, err = strconv.ParseInt(cols[0], 10, 32)
        if err != nil { log.Fatal(cols[0], err) }
        dx := make([]float64, 784)
        var f float64
        for i := 1; i < 785; i++ {
            f, err = strconv.ParseFloat(cols[i], 32)
            if err != nil { log.Fatal(cols[i], err) }
            dx[i - 1] = f
        }
        m := mat64.NewDense(len(dx), 1, dx)
        if n > 9 || f > 10000 || m == nil {
            log.Fatalf("n=%d, f=%f, m=%#v", n, f, m)
        }
        dat = append(dat, &ITEM{x: m, y: int(n)})
    }
    return
}


// The sigmoid function.
// Scale, Inverse, Apply
func sigmoid(z *mat64.Dense) *mat64.Dense {
    //return 1.0 / (1.0 + math.Exp(-z))
    ret := mat64.NewDense(0, 0, nil)
    ret.Apply(func(x, y int, v float64) float64 {
                return 1.0 / (1.0 + math.Exp(-v))
              }, z)
    return ret
}


// Derivative of the sigmoid function.
func sigmoid_prime(z *mat64.Dense) *mat64.Dense {
    //return sigmoid(z) * (1 - sigmoid(z))
    s := sigmoid(z)
    m := mat64.NewDense(0, 0, nil)
    /*
    a := mat64.NewDense(0, 0, nil)
    a.Apply(func(x, y int, v float64)float64{return 1.0 - v}, s)
    m.MulElem(s, a)
    */
    m.Apply(func(x, y int, v float64)float64{
                return v * (1.0 - v)
            }, s)
    return m
}


func randyx(y, x int) *mat64.Dense {
    z := x * y
    dat := make([]float64, z)
    for i := 0; i < z; i++ {
        dat[i] = rand.Float64()
    }
    return mat64.NewDense(y, x, dat)
}


func dot(w, a *mat64.Dense) *mat64.Dense {
    r, _ := w.Dims()
    d := make([]float64, r)
    v := a.ColView(0)
    for i := 0; i < len(d); i++ {
        d[i] = mat64.Dot(w.RowView(i), v)
    }
    return mat64.NewDense(r, 1, d)
}


func argmax(a *mat64.Dense) int {
    f := a.At(0, 0)
    idx := 0
    r, c := a.Dims()
    for i := 0; i < r; i++ {
        for j := 0; j < c; j++ {
            t := a.At(i, j)
            if t > f {
                f = t
                idx = i * c + j
            }
        }
    }
    return idx
}


// http://stackoverflow.com/questions/12264789/shuffle-array-in-go
func shuffle(dat []*ITEM) {
    for i := range dat {
        j := rand.Intn(i + 1)
        dat[i], dat[j] = dat[j], dat[i]
    }
}


func zip(A, B []*mat64.Dense, f func(x,y *mat64.Dense)*mat64.Dense) []*mat64.Dense {
    Z := make([]*mat64.Dense, len(A))
    for i, a := range A {
        Z[i] = f(a, B[i])
    }
    return Z
}


func zeros(ts []*mat64.Dense) []*mat64.Dense {
    zs := make([]*mat64.Dense, len(ts))
    for i, t := range ts {
        r, c := t.Dims()
        zs[i] = mat64.NewDense(r, c, nil)
    }
    return zs
}


func reset(ts []*mat64.Dense) {
    for _, t := range ts {
        t.Reset()
    }
}


type Network struct {
    num_layers int
    sizes   []int
    biases  []*mat64.Dense
    weights []*mat64.Dense
}


func NewNetwork(sizes []int) *Network {
    nw := &Network{num_layers: len(sizes), sizes: sizes}
    nw.biases = make([]*mat64.Dense, nw.num_layers - 1)
    nw.weights = make([]*mat64.Dense, nw.num_layers - 1)
    for i := 0; i < nw.num_layers - 1; i++ {
        y := nw.sizes[i + 1]
        x := nw.sizes[i]
        nw.biases[i] = randyx(y, 1)
        nw.weights[i] = randyx(y, x)
    }
    return nw
}


func (nw *Network)feedforward (a *mat64.Dense) *mat64.Dense{
    /*
    for i := 0; i < len(nw.biases); i++ {
        s := mat64.NewDense(0, 0, nil)
        s.Add(dot(nw.weights[i], a), nw.biases[i])
        a = sigmoid(s)
    }*/
    for i, b := range nw.biases {
        s := mat64.NewDense(0, 0, nil)
        s.Add(dot(nw.weights[i], a), b)
        a = sigmoid(s)
    }
    return a
}

func (nw *Network)SGD(training_data []*ITEM, epochs, mini_batch_size int,
                      eta float64, test_data []*ITEM) {
    for j := 0; j < epochs; j++ {
        shuffle(training_data)
        for k :=0; k < len(training_data); k = k + mini_batch_size {
            nw.update_mini_batch(training_data[k:k+mini_batch_size], eta)
        }
        if test_data != nil {
            fmt.Printf("Epoch %02d: %d / %d\n",
                       j, nw.evaluate(test_data), len(test_data))
        } else {
            fmt.Printf("Epoch %02d complete", j)
        }
    }
}


func (nw *Network)mb_add(ns, dn []*mat64.Dense) []*mat64.Dense {
    z := make([]*mat64.Dense, len(ns))
    for i := range ns {
        x := mat64.NewDense(0, 0, nil)
        x.Add(ns[i], dn[i])
        z[i] = x
    }
    return z
}

func (nw *Network)mb_cal(p float64, ns, dn []*mat64.Dense) []*mat64.Dense {
    z := make([]*mat64.Dense, len(ns))
    for i := range dn {
        m := mat64.NewDense(0, 0, nil)
        m.Apply(func(r, c int, v float64)float64{
                    return ns[i].At(r, c) - p * v
                }, dn[i])
        z[i] = m
    }
    return z
}

func (nw *Network)update_mini_batch(mini_batch []*ITEM, eta float64) {
    nabla_b := zeros(nw.biases)
    nabla_w := zeros(nw.weights)
    for _, it  := range mini_batch {
        delta_nabla_b, delta_nabla_w := nw.backprop(it)
        nabla_b = nw.mb_add(nabla_b, delta_nabla_b)
        nabla_w = nw.mb_add(nabla_w, delta_nabla_w)
    }
    p := eta / float64(len(mini_batch))
    nw.weights = nw.mb_cal(p, nw.weights, nabla_w)
    nw.biases = nw.mb_cal(p, nw.biases, nabla_b)
}

func (nw *Network)backprop(it *ITEM) (dnb, dnw []*mat64.Dense) {
    return
}

func (nw *Network)evaluate(test_data []*ITEM) int {
    eq := 0
    for _, item := range test_data {
        if item.y == argmax(nw.feedforward(item.x)) { eq = eq + 1 }
    }
    return eq
}

func (nw *Network)cost_derivative(output_activations, y *mat64.Dense) *mat64.Dense{
    m := mat64.NewDense(0, 0, nil)
    m.Sub(output_activations, y)
    return m
}


func main() {
    /*
    load_one("trai_data.txt")
    load_one("vali_data.txt")
    load_one("test_data.txt")
    */
}
