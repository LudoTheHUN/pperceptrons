# pperceptrons

A Clojure library designed to implement a [pperceptron](http://www.igi.tugraz.at/psfiles/pdelta-journal.pdf) using core.matrix.

An alternative to traditional [neural networks](http://en.wikipedia.org/wiki/Artificial_neural_network) for [function approximation](http://en.wikipedia.org/wiki/Function_approximation).

A parallel perceptron (pperceptron or pp's for short), trained via the p-delta learning rule, can approximate a function from ℝⁿ to the range [-1.0 , 1.0]. It can do this to arbitrary accuracy given appropriate parameters.

Read the paper to learn more.

While epoch based learning utility functions are available, they all fall back to iterative on-line learning implemented in `train`.

All meta parameters (like learning rate) are auto tuned, no parameter tuning is required (when training with `train-seq-epochs`). You just have to create an appropriate pp for your task.

## Usage

add to project.clj

`[org.clojars.ludothehun/pperceptrons "0.1.0-SNAPSHOT"]`

require in the library:

```(ns yourns
  (:require [pperceptrons.core]))```

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
   2       ;inputsize    ;how wide is the input, for this example, we have an input of size 2
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
       (second (first input)))  ;output to train with for that input
```

but this is only one instance of training, over just 1 of the 4 training examples. It can takes 100's of epochs for the pp to settle to the intended answer

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

tast/pperceptrons/iris_pp_tests.clj has an axample of training over the [Iris](http://en.wikipedia.org/wiki/Iris_flower_data_set) data set, where high accuracy is achived.


Notes:

- There are no guarantees of convergence or generalisation. Unless you feed a pp contradictory data, it should make progress towards a better approximation, given the epsilon of error you specified
- A pp may be too small to learn a target function by default, you may need to set `:size-boost` option to > 1 for complicated datasets to improve accuracy.
- If you want to create a pp manually, have a look at the `make-resonable-pp` implementation and the underlying record.


## Feedback

Please let me know if you find this at all useful and please feedback on github.

## License

Copyright © 2014 Ludwik Grodzki

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
