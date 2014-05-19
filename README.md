# pperceptrons

A Clojure library designed to implement [pperceptrons](http://www.igi.tugraz.at/psfiles/pdelta-journal.pdf) using core.matrix.

Parallel perceptions (pperceptrons or pp's for short), via the p-delta learning rule, can approximate functions from R^n to the range [-1.0 , 1.0] to arbitrary accuracy given appropriate paramters.

Read the paper to learn more.

While epoch based utility functions are available, they all fall back to iterative on-line learning implemented in `train`.

WIP: Key meta parameters (like learning rate) are auto tuned, no paramter tuning is required.

## Usage

Lets say you want to train a pperceptron to solve for the XOR function. Craft your data into the shape `[[[input1 input2 ...] ouput] ...]` . eg:

```Clojure
(def input [
    [[-1.0  1.0] -1.0] [[ 1.0  1.0]  1.0]
    [[-1.0 -1.0]  1.0] [[ 1.0 -1.0] -1.0]
   ])
 ;; Input-output pairs of the XOR function.
```

Create a pperceptron

```Clojure
(def pp
 (make-resonable-pp
   2       ;inputsize    ;;how wide is the input, for this examples, we have an input of size 2
   0.501   ;epsilon      ;How accurate do you need to be. Use 0.501 for a binary pperceptron (which will return -1.0 or 1.0, when zerod? = false). Smaller epsilon will make the pp bigger internally.
   false   ;zerod?       ;true makes the number of intrnal perceptrons even, so it will be possible to respond with 0.0 as the output.
      ; & ops
   ))
;; ops options are:
;  :seed <number> ;;The random number seed used to generate the pperceptron, default 0
;  :size-boost <number> ;; How many times larger should the pp be internally then the default. Default is 1. >1 integer values will allow the pp to learn more complicated functions (with more inflection points)
;  :matrix-implementation <:keyword> ;; The core.matrix implementation the pp should use. Default :vectorz
```

In [thoery](http://www.igi.tugraz.at/psfiles/pdelta-journal.pdf), you can make epsilon as small as you want (but >0), to achieve arbitrary accuracy. You may run out of RAM or time however.

You can train the pp with

```Clojure
(train pp
       (ffirst input) ;input
       (second (first input)))  ;output
```

but this is only one instance of training, over just 1 of the 4 training examples. It can takes 100's of epochs for the pperceptron to settle to the intended answer

To train the pp over n-epochs of the data, use `train-seq-epochs`

```Clojure
(def pp-trained  (let [n-epochs 400]
                    (train-seq-epochs pp input n-epochs)))
```

The pperceptron should have learned the input data by now.

```Clojure
(read-out pp-trained [-1.0  1.0])   ;=> -1.0
(read-out pp-trained [ 1.0  1.0])   ;=>  1.0
(read-out pp-trained [-1.0 -1.0])   ;=>  1.0
(read-out pp-trained [ 1.0 -1.0])   ;=> -1.0
```

and indeed it has.

Note that it had to change each of it's pre training values (so it wasn't just 1 in 16 chance that it was 'correct' by luck after initializing). Use a different seed (or map the training over many seeds) to check for robustness.

```Clojure
(read-out pp [-1.0  1.0])   ;=>  1.0
(read-out pp [ 1.0  1.0])   ;=> -1.0
(read-out pp [-1.0 -1.0])   ;=> -1.0
(read-out pp [ 1.0 -1.0])   ;=>  1.0
;;all these are opposite of the target trained result.
```

You can ask for any value on the input side. In this case, we see that the pp-trained happens to be generalising well.

```Clojure
(read-out pp-trained [-1.8  1.7])  ;=> -1.0
(read-out pp-trained [ 1.8  1.7])  ;=>  1.0
(read-out pp-trained [-1.8 -1.7])  ;=>  1.0
(read-out pp-trained [ 1.8 -1.7])  ;=> -1.0

(read-out pp-trained [-0.8  0.7])  ;=>  1.0
(read-out pp-trained [ 0.8  0.7])  ;=> -1.0
(read-out pp-trained [-0.8 -0.7])  ;=> -1.0
(read-out pp-trained [ 0.8 -0.7])  ;=>  1.0

(read-out pp-trained [-0.9  0.9])  ;=>  1.0
(read-out pp-trained [ 0.9  0.9])  ;=> -1.0
(read-out pp-trained [-0.9 -0.9])  ;=> -1.0
(read-out pp-trained [ 0.9 -0.9])  ;=>  1.0
```

Internally, for this most trivial of examples, the pp is represented by 3 by 3 matrix of Double's.

Notes:

- There are no guarantees of convergence or generalisation. Unless you feed a pp contradictory data, it should make progress towards a better approximation.
- You may need to set :size-boost > 1 for complicated datasets to improve accuracy.
- If you want to create the pp paramters manually, have a look at the `make-resonable-pp` implementation.


## License

Copyright © 2014 Ludwik Grodzki

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
