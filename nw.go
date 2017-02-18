package main

import (
    "fmt"
    "sync"
    "github.com/gonum/floats"
    "github.com/gonum/matrix/mat64"
)


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
        nw.biases[i] = randyx(y, 1, 1)
        nw.weights[i] = randyx(y, x, 1)
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
    thrd := 2
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


