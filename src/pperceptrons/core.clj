(ns pperceptrons.core
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as m-ops]
            [clojure.core.matrix.implementations :as imp]))

;;TODO DONE clean out to tests
;;TODO doc strings
;;TODO think about dynamics of eta--learning-rate and gamma--margin-around-zero

;(m/set-current-implementation :persistent-vector)
;(m/set-current-implementation :vectorz)


;;Resources
;https://github.com/mikera/core.matrix
;https://github.com/mikera/core.matrix/blob/master/src/main/clojure/clojure/core/matrix/examples.clj
;https://github.com/clojure-numerics/core.matrix.stats/blob/develop/src/main/clojure/clojure/core/matrix/random.clj
;http://www.igi.tugraz.at/psfiles/pdelta-journal.pdf


#_(m/set-current-implementation :persistent-vector)
#_(m/set-current-implementation :vectorz)


(defn input->z--input-array
  "adds the extra normalization to the input" [input-vector]
     (m/array (conj input-vector -1.0)))


(defn perceptron-f [a--perceptron-weight-vector z--input-vector]
     (let [mmulresult (m/scalar (m/mmul a--perceptron-weight-vector z--input-vector))]    ;;had to add m/scalar here to allow other matrix implementations
       (if (pos? mmulresult) 1.0 -1.0)))
;; NOTE we re-implement the above in the learning steps so that we don't compute the same a.z values more then once

;; NOTE pperceptron is made of many a--perceptron-weight-vectors


(defn total-pperceptron [pperceptron z--input-vector]
  (let [perceptron_value_fn (fn [perceptron] (perceptron-f perceptron z--input-vector))]
    (reduce +
            (doall (map perceptron_value_fn  (m/slices pperceptron))))))


(defn sp--squashing-function [pp-total rho--squashing-parameter]
     (cond
         (> pp-total rho--squashing-parameter)
           1.0
         (< pp-total (- rho--squashing-parameter))
           -1.0
         :else
         (/ pp-total rho--squashing-parameter)))


(defn b-sp--binary-squashing-function [pp-total]
    (if (pos? pp-total ) 1.0 -1.0))


(defn pp-output [pperceptron input rho--squashing-parameter]
              (sp--squashing-function (total-pperceptron pperceptron    (input->z--input-array input))   rho--squashing-parameter))


(defn perceptron-f [a--perceptron-weight-vector z--input-vector]
     (let [mmulresult (m/scalar (m/mmul a--perceptron-weight-vector z--input-vector))]
       (if (pos? mmulresult) 1.0 -1.0)))


(defn scaling-to-one-fn [perceptron eta--learning-rate]
   (m-ops/* perceptron (* -1.0 eta--learning-rate (- (m/length-squared perceptron) 1.0))))


(defn pdelta-update-with-margin [pperceptron matrix-implementation input target-output epsilon rho--squashing-parameter eta--learning-rate mu-zeromargin-importance gamma--margin-around-zero]
  (let [z--input-vector       (input->z--input-array input)
       perceptron_value_fn    (fn [perceptron] (m/scalar (m/mmul perceptron z--input-vector)))   ;;had to add m/scalar here to allow other matrix implementations
       per-perceptron-totals  (doall (map perceptron_value_fn  (m/slices pperceptron)))
       out                    (sp--squashing-function (reduce + (doall (map #(if (pos? (m/scalar %)) 1.0 -1.0) per-perceptron-totals))) rho--squashing-parameter)
       ]
    (m/matrix matrix-implementation
              (doall (map  (fn [perceptron perceptron_value]
                       (cond
                         (and (> out (+ target-output epsilon)) (pos? perceptron_value))
                           (m-ops/+ perceptron (scaling-to-one-fn perceptron eta--learning-rate) (m-ops/* z--input-vector -1.0 eta--learning-rate))
                         (and (< out (- target-output epsilon)) (neg? perceptron_value))
                           (m-ops/+ perceptron (scaling-to-one-fn perceptron eta--learning-rate) (m-ops/* z--input-vector       eta--learning-rate))
                         (and (<= out (+ target-output epsilon)) (pos? perceptron_value) (< perceptron_value gamma--margin-around-zero))
                           (m-ops/+ perceptron (scaling-to-one-fn perceptron eta--learning-rate) (m-ops/* z--input-vector  mu-zeromargin-importance  eta--learning-rate))
                         (and (>= out (- target-output epsilon))  (neg? perceptron_value) (< (* -1.0 gamma--margin-around-zero) perceptron_value ))
                           (m-ops/+ perceptron (scaling-to-one-fn perceptron eta--learning-rate) (m-ops/* z--input-vector -1.0  mu-zeromargin-importance  eta--learning-rate))
                         :else
                           (m-ops/+ perceptron (scaling-to-one-fn perceptron eta--learning-rate) )))
                    pperceptron
                    per-perceptron-totals)))))


(defprotocol PPperceptron
  "Protocol for working with paralel perceptrons, pperceptron-record"
  (read-out     [pp input]  "returns the output the pperceptron for the given input")
  (train        [pp input output] "trains the paralel perceptron on one input-output example")
  (train-seq    [pp input-output-seq] "trains the paralel perceptron on a sequence of input examples with output values, shaped as [[inputs output]...]. This is epoch based training, one epoch only")
  (train-seq-epochs    [pp input-output-seq n-epochs] "trains the paralel perceptron on a sequence of input examples with output values, shaped as [[inputs output]...]. This is epoch based training, n-epochs")
  (anneal-eta   [pp] "anneal eta, the learning rate"))


(defrecord pperceptron-record
  [pperceptron               ;; pperceptron ; the matrix holding the paralel perceptron weights which is as wide as the input +1 and as high as the number of perceptrons, n.
   matrix-implementation     ;; As supported by core.matrix, tested against  :persistent-vector and :vectorz
   n                         ;; n ; the total number of perceptrons in the pperceptron
   pwidth                    ;; size of each perceptron , width of pperceptron, +1 to size of input
   eta--learning-rate        ;; eta--learning-rate ; The learing rate. Typically 0.01 or less. Should be annealed.
   epsilon                   ;; epsilon ; how accureate we want to be, must be > 0
   rho--squashing-parameter  ;; rho--squashing-parameter ; An int. If set to 1, will force the pp to have binary output (-1,+1) in n is odd. Can be at most n. Typically set to  (/ 1 (* 2 epsilon))
   mu-zeromargin-importance  ;; mu-zeromargin-importance ; The zero margin parameter. Typically 1.
   gamma--margin-around-zero ;; gamma--margin-around-zero ; Margin around zero of the perceptron. Needs to be controlled for best performance, else set between 0.1 to 0.5
   ])


;;TODO hookup training function
(extend-protocol PPperceptron
  pperceptron-record
  (read-out [pp input]
     (pp-output (:pperceptron pp) input (:rho--squashing-parameter pp)))
  (train [pp input output]
      (assoc pp :pperceptron
        (pdelta-update-with-margin
           (:pperceptron pp)
           (:matrix-implementation pp)
           input
           output ;; target-output
           (:epsilon pp)
           (:rho--squashing-parameter pp)
           (:eta--learning-rate pp)
           (:mu-zeromargin-importance pp)
           (:gamma--margin-around-zero pp)
          )))
  (train-seq [pp input-output-seq]
        (reduce (fn [xs [in out]] (train xs in out)) pp input-output-seq))
  (train-seq-epochs [pp input-output-seq n-epochs]
             (reduce (fn [xs times] (train-seq xs input-output-seq)) pp (range n-epochs)))
  (anneal-eta [pp] (update-in pp [:eta--learning-rate] (fn [x] (* x 0.999)) ))  ;;WIP there is a specific non trivial algo for this based on error function
)


(defn uniform-dist-matrix-center-0
  "Returns an array of random samples from a uniform distribution on [0,1)
   Size may be either a number of samples or a shape vector."
  ([size seed] (uniform-dist-matrix-center-0 (m/current-implementation) size seed))
  ([matrix-implementation size seed]
    (let [size (if (number? size) [size] size)
          rnd  (java.util.Random. seed)]
      (m/compute-matrix matrix-implementation size
        (fn [& ixs]
          (- (* 2.0 (.nextDouble rnd)) 1.0))))))

#_(uniform-dist-matrix-center-0 :vectorz [3 3] 42)
#_(imp/get-implementation-key (uniform-dist-matrix-center-0 :vectorz [3] 42))


;;TODO DONE constructing close to length one perceptrons from the start...
;;TODO DONE take the random output and assuming it's a pperceptron, scale each perceptron towards length 1.0

(defn scale-to-size-one
 ([pperceptron] (scale-to-size-one (imp/get-implementation-key pperceptron) pperceptron ))
 ([matrix-implementation pperceptron]
   (m/matrix matrix-implementation (map m/normalise (m/slices pperceptron)))))


(defn make-resonable-pp
 ([inputsize epsilon zerod? seed] (make-resonable-pp inputsize epsilon zerod? seed 1 :vectorz))
 ([inputsize epsilon zerod? seed size-boost] (make-resonable-pp inputsize epsilon zerod? seed size-boost :vectorz))
 (                    [inputsize  ;;do a (count input)
                       epsilon    ;;less then 0.5, nice numers here are eg: 0.25, 0.1
                       zerod?     ;;if true, n, number of perceptrons, will be even hence zero will be a possible output.
                       seed       ;;some int of your choosing
                       size-boost
                       matrix-implementation] ;;How many times should the pp be bigger then recomended
 (let [pwidth      (+ inputsize 1)
       n-prez (int (* (/ 2 epsilon) size-boost))
       n     (cond (and zerod? (even? n-prez))
                     n-prez
                   (and zerod? (odd? n-prez))
                     (+ 1 n-prez)  ;to make it even
                   (and (not zerod?) (odd? n-prez))
                     n-prez
                   (and (not zerod?) (even? n-prez))
                     (+ 1 n-prez)
                   :else :this_should_never_happen!)
       rho-wip  (int (/ 1 (* 1 epsilon)))  ;;paper says 2, but this does not reconcile with example on page 6.
       rho      (if (= rho-wip 0) 1 rho-wip)]
  (new pperceptron-record
  ; (scale-to-size-one (uniform-dist-matrix-center-0 matrix-implementation [n pwidth] seed))   ;; pperceptron ; the matrix holding the paralel perceptron weights which is as wide as the input +1 and as high as the number of perceptrons, n.
   (uniform-dist-matrix-center-0 matrix-implementation [n pwidth] seed)
   matrix-implementation     ;; As supported by core.matrix, tested against  :persistent-vector and :vectorz
   n                         ;; n ; the total number of perceptrons in the pperceptron
   pwidth                    ;; size of each perceptron , width of pperceptron, +1 to size of input
   0.01                      ;; eta--learning-rate ; The learing rate. Typically 0.01 or less. Should be annealed or be dynamicall updated based on error function
   epsilon                   ;; epsilon ; how accureate we want to be, must be > 0
   rho                       ;; rho--squashing-parameter ; An int. If set to 1, will force the pp to have binary output (-1,+1) in n is odd. Can be at most n. Typically set to  (/ 1 (* 2 epsilon))
   1.0                       ;; mu-zeromargin-importance ; The zero margin parameter. Typically 1.
   0.01   ;was 0.5            ;; gamma--margin-around-zero ; Margin around zero of the perceptron. Needs to be controlled for best performance, else set between 0.1 to 0.5
  ))))
















