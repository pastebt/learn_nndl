package main

import (
    "os"
    "fmt"
    "log"
    "math"
    "math/rand"
    "sync"
    "bufio"
    "strconv"
    "strings"
    "github.com/gonum/floats"
    "github.com/gonum/blas/cgo"
    "github.com/gonum/blas/blas64"
    "github.com/gonum/matrix/mat64"
)


type ITEM struct {
/////////////////////+build cblas
    x, yv *mat64.Dense
    yi int
}


func load_one(infn string) (dat []*ITEM, err error) {

    blas64.Use(cgo.Implementation{})

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
        v := mat64.NewDense(10, 1, nil)
        y := int(n)
        v.Set(y, 0, 1.0)
        dat = append(dat, &ITEM{x: m, yv: v, yi: y})
    }
    return
}


// The sigmoid function.
func sigmoid_(z *mat64.Dense) *mat64.Dense {
    //return 1.0 / (1.0 + math.Exp(-z))
    ret := mat64.NewDense(0, 0, nil)
    ret.Apply(func(x, y int, v float64) float64 {
                return 1.0 / (1.0 + math.Exp(-v))
              }, z)
    return ret
}

func sigmoid64(fs []float64) {
    for i, f := range fs {
        fs[i] = 1.0 / (1.0 + math.Exp(-f))
    }
}
func sigmoid(z *mat64.Dense) *mat64.Dense {
    s := mat64.DenseCopyOf(z)
    sigmoid64(s.RawMatrix().Data)
    return s
}


// Derivative of the sigmoid function.
func sigmoid_prime_(z *mat64.Dense) *mat64.Dense {
    //return sigmoid(z) * (1 - sigmoid(z))
    s := sigmoid(z)
    m := mat64.NewDense(0, 0, nil)
    m.Apply(func(x, y int, v float64)float64{
                return v * (1.0 - v)
            }, s)
    return m
}

func sigmoid_prime(z *mat64.Dense) *mat64.Dense {
    s := mat64.DenseCopyOf(z)
    fs := s.RawMatrix().Data
    sigmoid64(fs)
    for i, f := range fs {
        fs[i] = f * (1.0 - f)
    }
    return s
}


func randyx(y, x int) *mat64.Dense {
    z := x * y
    dat := make([]float64, z)
    for i := 0; i < z; i++ {
        //dat[i] = rand.Float64() * 2.0 - 1.0
        dat[i] = rand.NormFloat64()
    }
    return mat64.NewDense(y, x, dat)
}


func npdot(w, a mat64.Matrix) *mat64.Dense {
    z := mat64.NewDense(0, 0, nil)
    z.Product(w, a)
    return z
}


func argmax(a *mat64.Dense) int {
    i := floats.MaxIdx(a.RawMatrix().Data)
    return i
}


// http://stackoverflow.com/questions/12264789/shuffle-array-in-go
func shuffle(dat []*ITEM) {
    for i := range dat {
        j := rand.Intn(i + 1)
        dat[i], dat[j] = dat[j], dat[i]
    }
}


func zeros(ts []*mat64.Dense) []*mat64.Dense {
    zs := make([]*mat64.Dense, len(ts))
    for i, t := range ts {
        r, c := t.Dims()
        zs[i] = mat64.NewDense(r, c, nil)
    }
    return zs
}


func add(dst, src *mat64.Dense) (z *mat64.Dense) {
    rd, cd := dst.Dims()
    rs, cs := src.Dims()
    if rd == rs && cd == cs {
        z = mat64.NewDense(0, 0, nil)
        z.Add(dst, src)
        return
    }
    if (rd == 1 || cd == 1) {
        src, dst = dst, src
        rd, cd, rs, cs = rs, cs, rd, cd
    }
    if rs == 1 && cs == cd {
        vs := src.RowView(0).RawVector()
        z = mat64.DenseCopyOf(dst)
        for r := 0; r < rd; r++ {
            vd := z.RowView(r).RawVector()
            blas64.Axpy(cd, 1, vs, vd)
        }
        return
    } else if cs == 1 && rd == rs {
        vs := src.ColView(0).RawVector()
        z = mat64.DenseCopyOf(dst)
        for c := 0; c < cd; c++ {
            vd := z.ColView(c).RawVector()
            blas64.Axpy(rd, 1, vs, vd)
        }
        return
    }
    return
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


func (nw *Network)feedforward(a *mat64.Dense) *mat64.Dense{
    for i, b := range nw.biases {
        a = sigmoid(add(npdot(nw.weights[i], a), b))
    }
    return a
}

func (nw *Network)SGD(training_data []*ITEM, epochs, mini_batch_size int,
                      eta float64, test_data []*ITEM) {
    for j := 0; j < epochs; j++ {
        shuffle(training_data)
        for k :=0; k < len(training_data); k = k + mini_batch_size {
            nw.update_mini_batch_m(training_data[k:k+mini_batch_size], eta)
        }
        if test_data != nil {
            fmt.Printf("Epoch %02d: %d / %d\n",
                       j, nw.evaluate(test_data), len(test_data))
        } else {
            fmt.Printf("Epoch %02d complete\n", j)
        }
    }
}


func (nw *Network)mb_add(ns, dn []*mat64.Dense) []*mat64.Dense {
    z := make([]*mat64.Dense, len(ns))
    for i := range ns {
        z[i] = add(ns[i], dn[i])
    }
    return z
}

func (nw *Network)mb_cal_(p float64, ns, dn []*mat64.Dense) []*mat64.Dense {
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
func (nw *Network)mb_cal(p float64, ns, dn []*mat64.Dense) []*mat64.Dense {
    z := make([]*mat64.Dense, len(ns))
    for i := range dn {
        m := mat64.DenseCopyOf(ns[i])
        fs := m.RawMatrix().Data
        ds := dn[i].RawMatrix().Data
        for j, f := range fs {
            fs[j] = f - p * ds[j]
        }
        z[i] = m
    }
    return z
}
func (nw *Network)mb_cal__(p float64, ns, dn []*mat64.Dense) []*mat64.Dense {
    z := make([]*mat64.Dense, len(ns))
    for i := range dn {
        r, c := ns[i].Dims()
        m := mat64.NewDense(r, c, nil)
        dst := m.RawMatrix().Data
        floats.AddScaledTo(dst, ns[i].RawMatrix().Data, -p, dn[i].RawMatrix().Data)
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

func (nw *Network)update_mini_batch_m(mini_batch []*ITEM, eta float64) {
    lm := len(mini_batch)
    var wg1, wg2 sync.WaitGroup
    nabla_b := zeros(nw.biases)
    nabla_w := zeros(nw.weights)
    itch := make(chan *ITEM, 100)
    rbch := make(chan []*mat64.Dense, 100)
    rwch := make(chan []*mat64.Dense, 100)
    go func(){
        for _, it  := range mini_batch { itch <- it }
        close(itch)
    }()
    thrd := 4
    wg1.Add(thrd)
    wg2.Add(2)
    for w := 0; w < thrd; w++ {
        go func() {
            for it := range itch {
                delta_nabla_b, delta_nabla_w := nw.backprop(it)
                //rtch <- [][]*mat64.Dense{delta_nabla_b, delta_nabla_w}
                rbch <- delta_nabla_b
                rwch <- delta_nabla_w
            }
            wg1.Done()
        }()
    }
    go func() {
        for delta_nabla_b := range rbch {
            nabla_b = nw.mb_add(nabla_b, delta_nabla_b)
        }
        wg2.Done()
    }()
    go func() {
        for delta_nabla_w := range rwch {
            nabla_w = nw.mb_add(nabla_w, delta_nabla_w)
        }
        wg2.Done()
    }()
    wg1.Wait()
    close(rbch)
    close(rwch)
    wg2.Wait()

    p := eta / float64(lm)
    nw.weights = nw.mb_cal(p, nw.weights, nabla_w)
    nw.biases = nw.mb_cal(p, nw.biases, nabla_b)
}

func (nw *Network)backprop(it *ITEM) (nabla_b, nabla_w []*mat64.Dense) {
    nabla_b = zeros(nw.biases)
    nabla_w = zeros(nw.weights)
    // feedforward
    activation := it.x
    // list to store all the activations, layer by layer
    activations := []*mat64.Dense{it.x}
    // list to store all the z vectors, layer by layer
    zs := make([]*mat64.Dense, 0, 10)
    for i := range nw.biases {
        z := mat64.NewDense(0, 0, nil)
        z.Add(npdot(nw.weights[i], activation), nw.biases[i])
        zs = append(zs, z)
        activation = sigmoid(z)
        activations = append(activations, activation)
    }
    // backward pass
    delta := mat64.NewDense(0, 0, nil)
    delta.MulElem(nw.cost_derivative(activations[len(activations) - 1], it.yv),
                  sigmoid_prime(zs[len(zs) - 1]))
    nabla_b[len(nabla_b) - 1] = delta
    nabla_w[len(nabla_w) - 1] = npdot(delta, activations[len(activations) - 2].T())
    for l := 2; l < nw.num_layers; l++ {
        z := zs[len(zs) - l]
        sp := sigmoid_prime(z)
        d := mat64.NewDense(0, 0, nil)
        d.MulElem(npdot(nw.weights[len(nw.weights) - l + 1].T(), delta), sp)
        delta = d
        nabla_b[len(nabla_b)-l] = delta
        nabla_w[len(nabla_w)-l] = npdot(delta, activations[len(activations)-l-1].T())
    }
    return
}

func (nw *Network)evaluate(test_data []*ITEM) int {
    eq := 0
    for _, item := range test_data {
        if item.yi == argmax(nw.feedforward(item.x)) { eq = eq + 1 }
    }
    return eq
}

func (nw *Network)cost_derivative(output_activations, y *mat64.Dense) *mat64.Dense{
    m := mat64.NewDense(0, 0, nil)
    m.Sub(output_activations, y)
    return m
}


func main() {
    trai, err := load_one("trai_data.txt")
    if err != nil { log.Fatal(err) }
    //_, err := load_one("vali_data.txt")
    //if err != nil { log.Fatal(err) }
    test, err := load_one("test_data.txt")
    if err != nil { log.Fatal(err) }
    n := NewNetwork([]int{784, 30, 10})
    n.SGD(trai, 30, 10, 4.0, test)
}
