(ns pperceptrons.core
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as m-ops]))


;;Resources
;https://github.com/mikera/core.matrix
;https://github.com/mikera/core.matrix/blob/master/src/main/clojure/clojure/core/matrix/examples.clj
;https://github.com/clojure-numerics/core.matrix.stats/blob/develop/src/main/clojure/clojure/core/matrix/random.clj


;http://www.igi.tugraz.at/psfiles/pdelta-journal.pdf

(def a (m/array [-1 2 -0.1]))

(def b (m/array [0.1 0.3 -0.4]))

(class a)
(m/shape a)

(m/mmul a b)


(defn input->z--input-vector [input-vector]
     (m/array (conj input-vector -1.0))
  )

(class -1.0)
(input->z--input-vector [1.0 2.0])




(defn perceptron-f [a--perceptron-weight-vector z--input-vector]
     (let [mmulresult (m/mmul a--perceptron-weight-vector z--input-vector)]
       (if (pos? mmulresult) 1.0 -1.0))
  )

;; we will need to re-implement the above in the learning steps so that we don't compute the same values more then once

(perceptron-f a b)


;pperceptron is made of many a--perceptron-weight-vectors

(def pperceptron
  ;;a pp with 3 perceptrons, taken in 3dim input, -1 in last dim for the threshold.
  (m/matrix
   [[-4.0 0.3 -0.3 0.5]
   [0.1 -0.2 -0.4 -0.6]
   [0.2 -0.1 -0.7 -0.6]]))

(class pperceptron)

(defn total-pperceptron [pperceptron z--input-vector]
  (let [perceptron_value_fn (fn [perceptron] (perceptron-f perceptron z--input-vector))]
    (reduce +
            (map perceptron_value_fn  (m/slices pperceptron)))
  ))


(total-pperceptron pperceptron  [-0.2 -0.2 10.5 -1.0])
(total-pperceptron pperceptron  [-0.2 -0.2 -10.5 -1.0])
(total-pperceptron pperceptron  [-0.2 -0.2 1.5 -1.0])




(defn sp--squashing-function [pp-total rho--squashing-parameter]
     (cond
         (> pp-total rho--squashing-parameter)
           1.0
         (< pp-total (- rho--squashing-parameter))
           -1.0
         :else
         (/ pp-total rho--squashing-parameter)
   ))


(defn b-sp--binary-squashing-function [pp-total]
    (if (pos? pp-total ) 1.0 -1.0))



(defn pp-output [pperceptron input rho--squashing-parameter]
              (sp--squashing-function (total-pperceptron pperceptron    (input->z--input-vector input))   rho--squashing-parameter)
  )




(pp-output pperceptron [-0.2 0.2 0.3] 3)
(pp-output pperceptron [-10.2 0.2 0.6] 10)


(def epsilon 0.3)   ;;how accureate we want to be, must be > 0

(def rho--squashing-parameter 10.0)  ;; whole numbers, 1 means a binary pperceptron

(defn f-abs [n]
 (cond
   (neg? n) (- n)
   :else n))


(def eta--learning-rate 0.1)


(defn perceptron-f [a--perceptron-weight-vector z--input-vector]
     (let [mmulresult (m/mmul a--perceptron-weight-vector z--input-vector)]
       (if (pos? mmulresult) 1.0 -1.0))
  )


(defn pdelta-update [pperceptron  input target-output epsilon rho--squashing-parameter eta--learning-rate]
   (let [z--input-vector (input->z--input-vector input)

         perceptron_value_fn (fn [perceptron] (m/mmul perceptron z--input-vector))
         per-perceptron-totals  (map perceptron_value_fn  (m/slices pperceptron))
         out  (sp--squashing-function (reduce + (map #(if (pos? %) 1.0 -1.0) per-perceptron-totals)) rho--squashing-parameter)
         out-vs-train-abs (f-abs (- out target-output))
         ]

       (cond
         (<= out-vs-train-abs epsilon)   ;;pp gave correct enough answer, so nothing to do
            pperceptron
         (> out (+ target-output epsilon))   ;;pp is producing too higher result
         ; ;:pperceptron_lower
            (m/matrix (map  (fn [perceptron perceptron_value]
                               (if (pos? perceptron_value)
                                   (m-ops/- perceptron (m-ops/* z--input-vector eta--learning-rate))
                                   perceptron))
                            pperceptron
                            per-perceptron-totals))
         (< out (- target-output epsilon))   ;;pp is producing too lower result
         ; :pperceptron_higer
            (m/matrix (map  (fn [perceptron perceptron_value]
                               (if (neg? perceptron_value)
                                   (m-ops/+ perceptron (m-ops/* z--input-vector eta--learning-rate))
                                   perceptron))
                            pperceptron
                            per-perceptron-totals))

           :else
             pperceptron)
     ;;per-perceptron-totals

  ))



(def input [-1.1 -3 0.3])
(m-ops/* input eta--learning-rate )


(pdelta-update pperceptron input 0.2 0.01 10.0 0.1)




(def gamma--margin-around-zero 0.1)


(defn scaling-adela-fn [perceptron eta--learning-rate]
   (m-ops/* perceptron (* -1.0 eta--learning-rate (- (m/length-squared perceptron) 1.0)))
  )

(scaling-adela-fn [0.2 0.0 0.4] 0.01)


;(defn perceptron-f-amount [a--perceptron-weight-vector z--input-vector]
;     (m/mmul a--perceptron-weight-vector z--input-vector))

;(defn perceptron-f-vote [mmulresult]
;     (if (pos? mmulresult) 1.0 -1.0))

(defn pdelta-update-with-margin-BUGGED [pperceptron  input target-output epsilon rho--squashing-parameter eta--learning-rate mu-zeromargin-importance gamma--margin-around-zero]
   (let [z--input-vector (input->z--input-vector input)

         perceptron_value_fn (fn [perceptron] (m/mmul perceptron z--input-vector))
         per-perceptron-totals  (map perceptron_value_fn  (m/slices pperceptron))
         out  (sp--squashing-function (reduce + (map #(if (pos? %) 1.0 -1.0) per-perceptron-totals)) rho--squashing-parameter)
       ;  out-vs-train-abs (f-abs (- out target-output))
         ]
   ;;This is completely wrong!!! We need to go over each slice, ie: perceptron, and cond within that context, else we will miss some of the updates.

     ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; Needs a full correction!
       (cond
       ;  (<= out-vs-train-abs epsilon)   ;;pp gave correct enough answer, so nothing to do
       ;      (m/matrix (map  (fn [perceptron perceptron_value]  (m-ops/+ perceptron (scaling-adela-fn perceptron eta--learning-rate) )  )
       ;                     pperceptron
       ;                     per-perceptron-totals))
         (> out (+ target-output epsilon))   ;;pp is producing too higher result
         ; pperceptron must be lowered
            (m/matrix (map  (fn [perceptron perceptron_value]
                               (if (pos? perceptron_value)
                                   (m-ops/+ perceptron (scaling-adela-fn perceptron eta--learning-rate) (m-ops/* z--input-vector -1.0 eta--learning-rate))
                                   (m-ops/+ perceptron (scaling-adela-fn perceptron eta--learning-rate) ) ))
                            pperceptron
                            per-perceptron-totals))
         (< out (- target-output epsilon))   ;;pp is producing too lower result
         ; pperceptron must be made to produce higher results
            (m/matrix (map  (fn [perceptron perceptron_value]
                               (if (neg? perceptron_value)
                                   (m-ops/+ perceptron (scaling-adela-fn perceptron eta--learning-rate) (m-ops/* z--input-vector       eta--learning-rate))
                                   (m-ops/+ perceptron (scaling-adela-fn perceptron eta--learning-rate) ) ))
                            pperceptron
                            per-perceptron-totals))

         (<= out (+ target-output epsilon) )   ;;output a bit above target ... increase all perceptrons that are close to zero
             (m/matrix (map  (fn [perceptron perceptron_value]
                               (if (and (pos? perceptron_value) (< perceptron_value gamma--margin-around-zero) )
                                   (m-ops/+ perceptron (scaling-adela-fn perceptron eta--learning-rate) (m-ops/* z--input-vector  mu-zeromargin-importance  eta--learning-rate))
                                   (m-ops/+ perceptron (scaling-adela-fn perceptron eta--learning-rate) ) ))
                            pperceptron
                            per-perceptron-totals))
         (>= out (- target-output epsilon) )   ;;output a bit above target ... increase all perceptrons that are close to zero
             (m/matrix (map  (fn [perceptron perceptron_value]
                               (if (and (neg? perceptron_value) (< (* -1.0 gamma--margin-around-zero) perceptron_value ) )  ;;WIP factor out so its done once
                                   (m-ops/+ perceptron (scaling-adela-fn perceptron eta--learning-rate) (m-ops/* z--input-vector -1.0  mu-zeromargin-importance  eta--learning-rate))
                                   (m-ops/+ perceptron (scaling-adela-fn perceptron eta--learning-rate) ) ))
                            pperceptron
                            per-perceptron-totals))
        :else
                          (m/matrix (map  (fn [perceptron perceptron_value]  (m-ops/+ perceptron (scaling-adela-fn perceptron eta--learning-rate) )  )
                            pperceptron
                            per-perceptron-totals)))
     ;;per-perceptron-totals
     ;out
  ))




(defn pdelta-update-with-margin [pperceptron  input target-output epsilon rho--squashing-parameter eta--learning-rate mu-zeromargin-importance gamma--margin-around-zero]
   (let [z--input-vector (input->z--input-vector input)

         perceptron_value_fn (fn [perceptron] (m/mmul perceptron z--input-vector))
         per-perceptron-totals  (map perceptron_value_fn  (m/slices pperceptron))
         out  (sp--squashing-function (reduce + (map #(if (pos? %) 1.0 -1.0) per-perceptron-totals)) rho--squashing-parameter)
       ;  out-vs-train-abs (f-abs (- out target-output))
         ]
   ;;This is completely wrong!!! We need to go over each slice, ie: perceptron, and cond within that context, else we will miss some of the updates.

     ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; Needs a full correction!
            (m/matrix (map  (fn [perceptron perceptron_value]
                               (cond
                                 (and (> out (+ target-output epsilon)) (pos? perceptron_value))
                                   (m-ops/+ perceptron (scaling-adela-fn perceptron eta--learning-rate) (m-ops/* z--input-vector -1.0 eta--learning-rate))
                                 (and (< out (- target-output epsilon)) (neg? perceptron_value))
                                   (m-ops/+ perceptron (scaling-adela-fn perceptron eta--learning-rate) (m-ops/* z--input-vector       eta--learning-rate))
                                 (and (<= out (+ target-output epsilon)) (pos? perceptron_value) (< perceptron_value gamma--margin-around-zero))
                                   (m-ops/+ perceptron (scaling-adela-fn perceptron eta--learning-rate) (m-ops/* z--input-vector  mu-zeromargin-importance  eta--learning-rate))
                                 (and (>= out (- target-output epsilon))  (neg? perceptron_value) (< (* -1.0 gamma--margin-around-zero) perceptron_value ))
                                   (m-ops/+ perceptron (scaling-adela-fn perceptron eta--learning-rate) (m-ops/* z--input-vector -1.0  mu-zeromargin-importance  eta--learning-rate))
                                 :else
                                   (m-ops/+ perceptron (scaling-adela-fn perceptron eta--learning-rate) )))
                            pperceptron
                            per-perceptron-totals))

  ))


pperceptron
input
(pdelta-update-with-margin
    pperceptron
    input
    0.0   ;;; target-output
    0.01  ;;; epsilon
    3.0   ;;; rho--squashing-parameter
    0.01  ;;; eta--learning-rate
    0.1   ;;; mu-zeromargin-importance
    0.1   ;;; gamma--margin-around-zero
 )



(last (take 100
      (iterate (fn [x]
(;pdelta-update-with-margin
 pdelta-update-with-margin-BUGGED
    x
    input
    1.0   ;;; target-output
    0.01  ;;; epsilon
    3.0   ;;; rho--squashing-parameter
    0.01  ;;; eta--learning-rate
    1.0   ;;; mu-zeromargin-importance
    0.5   ;;; gamma--margin-around-zero
 )) pperceptron)))

[[-4.0 0.3 -0.3 0.5] [0.1 -0.2 -0.4 -0.6] [0.2 -0.1 -0.7 -0.6]]

;;TODO think about dynamics of eta--learning-rate and gamma--margin-around-zero
  ;these may need a dynamic update schedule.
  ;will need container for all the other aspects of the pperceptron
  ;create them with protocols / records ?



(pp-output pperceptron input 3.0)

(m/join [1 2 3] [3 4 5])

(m/mmul [[0.1 0.2 0.3]
       [0.2 0.2 0.1]
       [0.3 0.1 -0.3]
       ]

      [1 2 3]
      )


(m/mmul [[0.1 0.2 0.3]
       [0.2 0.2 0.1]
       [0.3 0.1 -0.3]
       ]

      [[3 3 3]
       [2 2 2]
       [1 1 1]]
      )






