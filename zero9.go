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


func load_one(infn string) (err error) {
    fin, err := os.Open(infn)
    if err != nil {
        log.Fatal(err)
    }
    defer fin.Close()
    println(infn)
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
        if n > 9 || f > 10000 || m == nil { log.Fatalf("n=%d, f=%f, m=%#v", n, f, m) }
    }
    return
}


// The sigmoid function.
// Scale, Inverse, Apply
func sigmoid(z *mat64.Dense) *mat64.Dense {
    //return 1.0 / (1.0 + math.Exp(-z))
    a := mat64.NewDense(0, 0, nil)
    e := mat64.NewDense(0, 0, nil)
    o := mat64.NewDense(0, 0, nil)
    i := mat64.NewDense(0, 0, nil)
    a.Scale(-1, z)
    e.Exp(a)
    o.Apply(func(x, y int, v float64)float64{return 1.0 + v}, e)
    i.Inverse(o)
    return i
}


// Derivative of the sigmoid function.
func sigmoid_prime(z *mat64.Dense) *mat64.Dense {
    //return sigmoid(z) * (1 - sigmoid(z))
    m := mat64.NewDense(0, 0, nil)
    a := mat64.NewDense(0, 0, nil)
    s := sigmoid(z)
    a.Apply(func(x, y int, v float64)float64{return 1.0 - v}, s)
    m.Mul(s, a)
    return m
}


// The sigmoid function.
func sigmoid_(z float64) float64 {
    return 1.0 / (1.0 + math.Exp(-z))
}


// Derivative of the sigmoid function.
func sigmoid_prime_(z float64) float64 {
    return sigmoid_(z) * (1 - sigmoid_(z))
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
    //fmt.Printf("w = %#v\na = %#v\n", w, a)
    r, _ := w.Dims()
    //fmt.Printf("c=%#v\n", c)
    d := make([]float64, r)
    v := a.ColView(0)
    //fmt.Printf("v=%#v\n", v)
    for i := 0; i < len(d); i++ {
        d[i] = mat64.Dot(w.RowView(i), v)
    }
    //fmt.Printf("d = %#v\n", d)
    return mat64.NewDense(r, 1, d)
}


func argmax(a *mat64.Dense) int {
    f := a.At(0, 0)
    idx := 0
    r, c := a.Dims()
    if c != 1 { log.Fatal("argmax, c != 1") }
    for i := 0; i < r; i++ {
        t := a.At(i, 0)
        if t > f {
            f = t
            idx = i
        }
    }
    return idx
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


func (nw *Network)SGD() {}
func (nw *Network)update_mini_batch() {}
func (nw *Network)backprop() {}
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


func test() {
    /*
    load_one("trai_data.txt")
    load_one("vali_data.txt")
    load_one("test_data.txt")
    */
    //w := randyx(2, 2)
    //a := randyx(2, 1)
    w := mat64.NewDense(3, 2, []float64{1.682852  , -0.03961098,
                                        0.03793304,  0.42585802,
                                        0.74246289, -0.03209176})
    a := mat64.NewDense(2, 1, []float64{-0.1631285,
                                        -1.73508289})
    /* np.dot(w, a)
    array([[-0.20579278],
           [-0.74508692],
           [-0.06543499]]) */
    d := dot(w, a)
    fmt.Printf("d = %#v\n", d)
}


func main() {

}
