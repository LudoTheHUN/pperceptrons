(ns pperceptrons.core
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as m-ops]
            [clojure.core.matrix.implementations :as imp]
            [pperceptrons.frugalize :as frug]))


;;Resources
;https://github.com/mikera/core.matrix
;https://github.com/mikera/core.matrix/blob/master/src/main/clojure/clojure/core/matrix/examples.clj
;https://github.com/clojure-numerics/core.matrix.stats/blob/develop/src/main/clojure/clojure/core/matrix/random.clj
;http://www.igi.tugraz.at/psfiles/pdelta-journal.pdf



;;TODO DONE clean out to tests
;;TODO doc strings
;;TODO DONE think about dynamics of eta--learning-rate DONE and gamma--margin-around-zero DONE

;;TODO cleanup eta--learning-rate tunning
;;TODO emulate eta batch training in the online traning (now that we have a stable update model)
;;TODO add self logging as an option with in the pp... use dire lib?

;;TODO refactor tests to not inject into Pperceptron, but add to it
;;TODO Add diagnostics printing + addition of diagnostic data into pp with extra options, but not pp record properties


;;TODO add hyper-learning... learn meny differently seeded pps, prune and keep only the best one, be able to choose to boost if learning is not converging, this will remove one more hyper parameter.
  ;;This is over reach and a seperate project in it's own right.


;(m/set-current-implementation :persistent-vector)
;(m/set-current-implementation :vectorz)




;;;;;;;; PPperceptron protocol and record

(defprotocol PPperceptron
  "Protocol for working with paralel perceptrons, pperceptron-record"
  (read-out     [pp input]  "returns the output the pperceptron for the given input")
  (train        [pp input output] "trains the paralel perceptron on one input-output example")
  (train-seq    [pp input-output-seq] "trains the paralel perceptron on a sequence of input examples with output values, shaped as [[inputs output]...]. This is epoch based training, one epoch only")
  (train-seq-epochs    [pp input-output-seq n-epochs] "trains the paralel perceptron on a sequence of input examples with output values, shaped as [[inputs output]...]. This is epoch based training, n-epochs")
  (anneal-eta   [pp] "dumbly anneal eta, the learning rate, do not used, error function based integrated via :eta-tune parameter"))


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
   gamma--tunning-rate       ;; how quickly should gamma be tunned , good value is 0.1
   eta--auto-tune?           ;; true means we will auto tune eta--learning-rate based on the error function, this will make learning less performant (wall clock), but should increase learning rate in terms of epochs. Should also make training more robust
   ])


;;;;;;;; Building up p-delta learning rule

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
    (reduce + (doall (map perceptron_value_fn  (m/slices pperceptron))))))


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




;;TODO DONE refactor out the perceptron_value_fn and per-perceptron-totals so that the per-perceptron-totals can be used for gamma--margin-around-zero updates
  ;;TODO DONE need to move 'output' also so gamma can use it without recomputing it


(defn pdelta-update-with-margin [pperceptron matrix-implementation z--input-vector
                                 per-perceptron-totals output
                                 target-output epsilon rho--squashing-parameter eta--learning-rate mu-zeromargin-importance gamma--margin-around-zero]
  (m/matrix matrix-implementation
    (doall (map  (fn [perceptron perceptron_value]
                   (cond
                     (and (> output (+ target-output epsilon)) (pos? perceptron_value))
                       (m-ops/+ perceptron (scaling-to-one-fn perceptron eta--learning-rate) (m-ops/* z--input-vector -1.0 eta--learning-rate))
                     (and (< output (- target-output epsilon)) (neg? perceptron_value))
                       (m-ops/+ perceptron (scaling-to-one-fn perceptron eta--learning-rate) (m-ops/* z--input-vector       eta--learning-rate))
                     (and (<= output (+ target-output epsilon)) (pos? perceptron_value) (< perceptron_value gamma--margin-around-zero))
                       (m-ops/+ perceptron (scaling-to-one-fn perceptron eta--learning-rate) (m-ops/* z--input-vector  mu-zeromargin-importance  eta--learning-rate))
                     (and (>= output (- target-output epsilon))  (neg? perceptron_value) (< (* -1.0 gamma--margin-around-zero) perceptron_value ))
                       (m-ops/+ perceptron (scaling-to-one-fn perceptron eta--learning-rate) (m-ops/* z--input-vector -1.0  mu-zeromargin-importance  eta--learning-rate))
                     :else
                       (m-ops/+ perceptron (scaling-to-one-fn perceptron eta--learning-rate) )))
            pperceptron
            per-perceptron-totals))))





;;;;;;;; Computing error function

;;TODO epsilon learnig rate auto tunning via error function
(defn pp-error-function "compute the pp error function."
  [pp per-perceptron-totals output target-output]
   (let [epsilon (:epsilon pp)
         gamma--margin-around-zero (:gamma--margin-around-zero pp)
         mu-zeromargin-importance  (:mu-zeromargin-importance pp)
         mu*gamma (* mu-zeromargin-importance gamma--margin-around-zero)
         term1 (* 0.5 (reduce + (map (fn [perceptron] (let [sq-me (- (m/length-squared perceptron) 1.0)] (* sq-me sq-me) )) (m/slices (:pperceptron pp)))))]
  (+ term1
    (cond (> output (+ target-output epsilon))
             (reduce +
              (map (fn [perceptron_value] (+ (if (>= perceptron_value 0.0)
                                                 (+ mu*gamma perceptron_value) 0.0)
                                             (if (and (< (* -1.0 gamma--margin-around-zero) perceptron_value)
                                                      (< perceptron_value 0.0))
                                                 (+ mu*gamma (* mu-zeromargin-importance perceptron_value)) 0.0)))
                   per-perceptron-totals))
          (< output (- target-output epsilon))
             (reduce +
              (map (fn [perceptron_value] (+ (if (< perceptron_value 0.0)
                                                 (- mu*gamma perceptron_value) 0.0)
                                             (if (and (<= 0.0 perceptron_value)
                                                      (< perceptron_value gamma--margin-around-zero))
                                                 (- mu*gamma (* mu-zeromargin-importance perceptron_value)) 0.0)))
                   per-perceptron-totals))
          (<= (m/abs (- output target-output)) epsilon)
             (reduce +
              (map (fn [perceptron_value] (+ (if (and (< (* -1.0 gamma--margin-around-zero) perceptron_value)
                                                      (< perceptron_value 0.0))
                                                 (+ mu*gamma (* mu-zeromargin-importance perceptron_value)) 0.0)
                                             (if (and (<= 0.0 perceptron_value)
                                                      (< perceptron_value gamma--margin-around-zero))
                                                 (- mu*gamma (* mu-zeromargin-importance perceptron_value)) 0.0)))
                   per-perceptron-totals))
          :else 0.0))))


(defn pp-error-function-standalone "compute the pp error based on raw inputs."
 [pp input target-output]
 (let [output (read-out pp input)
       z--input-vector        (input->z--input-array input)
       perceptron_value_fn    (fn [perceptron] (m/scalar (m/mmul perceptron z--input-vector)))   ;;had to add m/scalar here to allow other matrix implementations
       per-perceptron-totals  (doall (map perceptron_value_fn  (m/slices (:pperceptron pp))))   ]
  (pp-error-function pp per-perceptron-totals output target-output)))


(defn epoch-errors "all error values over in input-outpet-seq" [pp input-output-seq]
  (map (fn [[input output]] (pp-error-function-standalone pp input output)) input-output-seq))




;;;;;;;; Auto tunning

(defn auto-tune---gamma--margin-around-zero [pp per-perceptron-totals output target-output]
   (let [gamma--margin-around-zero (:gamma--margin-around-zero pp)
         gamma--tunning-rate       (:gamma--tunning-rate pp)
         epsilon                   (:epsilon pp)
         Mlsit      (doall (map (fn [perceptron_value]
                             (cond (and (<= 0 perceptron_value) (< perceptron_value gamma--margin-around-zero)        (<= output (+ target-output epsilon)))
                                     :M+
                                   (and (< (* -1 gamma--margin-around-zero) perceptron_value) (< perceptron_value 0 ) (>= output (- target-output epsilon)))
                                     :M-
                                   :else :ignore))
                                per-perceptron-totals))
         M+count   (count (filter #{:M+} Mlsit))
         M-count   (count (filter #{:M-} Mlsit))
         Mmin       (* epsilon (:rho--squashing-parameter pp))
         Mmax       (* 4 Mmin)
         new-gamma--margin-around-zero  (+ gamma--margin-around-zero
                                            (* gamma--tunning-rate
                                               (:eta--learning-rate pp)
                                               (- Mmin (min Mmax (+ M+count M-count)))) )]
    (if (> new-gamma--margin-around-zero 0.9 )  ;;Not sure if this can be allowed to go high
          pp
          (assoc pp :gamma--margin-around-zero new-gamma--margin-around-zero))))


(defn eta-auto-tune [pp error-before error-after]
  ;;Learning rate adjustment strategy. Aim is to learn as fast as possible while sustaining decreases in error.
  (assoc pp :eta--learning-rate
    (let [eta--learning-rate (:eta--learning-rate pp)]
      ;;(println eta--learning-rate)
      (cond (and error-before error-after (< eta--learning-rate 0.000000002))
              0.001   ;;if we hit rock bottom, rock the boat a bit
            (and error-before error-after (> error-before error-after) (< eta--learning-rate 0.1))  ;what is reasonable maximum learning rate?
              (* eta--learning-rate 1.1)    ;1.1  ;;Error decreased, speed up learning a bit   ;WIP
            (and error-before error-after (< error-before error-after) (> eta--learning-rate 0.000000001))
              (* eta--learning-rate 0.5)  ;0.5 ;;Error increase, slow down learning;
            :else eta--learning-rate))))
;

;;PLAN compute the error-value of orgiginal pp given inputs output, target-output, but do so over some history or recent input outputs, typically whole epoch
      ;train the pp
      ;compute the error-value of trained pp given inputs output, target-output, this means recompute the per-perceptron-totals and output of the trained pp
      ;if error-value has decreased, * epsilon rate by 1.1, if error-value has increased , * epislon rate by 0.5





;;;;;;;; Training utilities


(defn shuffle-seeded
  "Return a random permutation of coll based on a seed"
  [coll seed]
  (let [al (java.util.ArrayList. coll)
        random (java.util.Random. seed)]
    (java.util.Collections/shuffle al random)
    (clojure.lang.RT/vector (.toArray al))))

;;(shuffle-seeded [1 2 3 4] 6)


(defn average [coll]  ;;TODO use core.matrix's average
  (/ (reduce + coll) (count coll)))




;;;;;;;; PPperceptron protocol implementation

(extend-protocol PPperceptron
  pperceptron-record
  (read-out [pp input]
     (pp-output (:pperceptron pp) input (:rho--squashing-parameter pp)))
  (train [pp input target-output]
      (let [z--input-vector        (input->z--input-array input)
            perceptron_value_fn    (fn [perceptron] (m/scalar (m/mmul perceptron z--input-vector)))   ;;had to add m/scalar here to allow other matrix implementations
            per-perceptron-totals  (doall (map perceptron_value_fn  (m/slices (:pperceptron pp))))
            output                 (sp--squashing-function (reduce + (doall (map #(if (pos? (m/scalar %)) 1.0 -1.0) per-perceptron-totals))) (:rho--squashing-parameter pp))
            pp-trained             (-> pp
                                    (auto-tune---gamma--margin-around-zero per-perceptron-totals output target-output)
                                    (assoc  :pperceptron
                                            (pdelta-update-with-margin
                                               (:pperceptron pp) (:matrix-implementation pp) z--input-vector
                                               per-perceptron-totals output
                                               target-output (:epsilon pp)  (:rho--squashing-parameter pp) (:eta--learning-rate pp) (:mu-zeromargin-importance pp)  (:gamma--margin-around-zero pp)))
                                    ;;TODO eta auto tunning shoul live here only
                                       )
            ]
        pp-trained))
  (train-seq [pp input-output-seq]
        (reduce (fn [xs [in out]] (train xs in out)) pp input-output-seq))
  (train-seq-epochs [pp input-output-seq n-epochs]
             (reduce (fn [xs times]
                       ;;TODO print end of epoch diagnostics here, make optional, add a loglevel property
                         (do #_(println "eta:" (format "%.12f" (:eta--learning-rate xs))
                                        "avg-epoch-error: " (format "%.12f" (:avg-epoch-error xs))
                                      )
                             ;;TODO epoch level eta tunning
                             ;;TODO refactor this reduction function out
                             ;;TODO implement auto tuning fully at the  train level
                             (if (:eta--auto-tune? xs)
                                   (let [trained-pp (train-seq xs  (shuffle-seeded input-output-seq times))
                                         avg-epoch-error (average (epoch-errors trained-pp input-output-seq))   ;do we want to do this fully streaming?... then we'd need a lookback horizon over which to average error, lift this out as stand alone so it could be used in streaming
                                         prev-avg-epoch-error (:avg-epoch-error trained-pp)
                                         ]
                                      (-> trained-pp
                                          (eta-auto-tune prev-avg-epoch-error avg-epoch-error)
                                          (conj {:avg-epoch-error avg-epoch-error})
                                     ))
                                   (train-seq xs  (shuffle-seeded input-output-seq times))
                               )
                         )
                       )
                     pp
                     (range n-epochs)))
  (anneal-eta [pp] (update-in pp [:eta--learning-rate] (fn [x] (* x 0.999)) ))  ;;WIP there is a specific non trivial algo for this based on error function
)




;;;;;;;; Helpers with creation of a pp

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


#_(uniform-dist-matrix-center-0 :vectorz [30 30] 42)
#_(imp/get-implementation-key (uniform-dist-matrix-center-0 :vectorz [3] 42))


(defn scale-to-size-one
  "rescale a pp matrix to size one"
 ([pperceptron] (scale-to-size-one (imp/get-implementation-key pperceptron) pperceptron ))
 ([matrix-implementation pperceptron]
   (m/matrix matrix-implementation (map m/normalise (m/slices pperceptron)))))


;;;;;;;; Utility to create pp

(defn make-resonable-pp "creates a pp with resonable defaults given user friendly parameters"
  ;;TODO much more documentation here about creating pp relavant to a problem
 ([inputsize epsilon zerod? & ops]
 (let [ {:keys [seed size-boost matrix-implementation eta--auto-tune? gamma--tunning-rate]
         :or  {seed 0
               size-boost 1
               matrix-implementation :vectorz
               eta--auto-tune? true
               gamma--tunning-rate 0.1}}  ops
       pwidth      (+ inputsize 1)
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
     0.01   ;was 0.5           ;; gamma--margin-around-zero ; Margin around zero of the perceptron. Is auto tuned. Needs to be controlled for best performance, else set between 0.01 to 0.5
     gamma--tunning-rate       ;; gamma--tunning-rate ; 0 means gamma will not be tunned
     eta--auto-tune?           ;; eta--auto-tune? Default true, chooses if we should auto tune the learnig rate
    )

   )))


#_(make-resonable-pp 3 0.5 true :seed 43 :size-boost 3)
#_(println (make-resonable-pp 3 0.05 true))
