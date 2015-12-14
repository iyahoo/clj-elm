(ns clj-elm.core
  (:require [clj-elm.data :refer [read-dataset normalize]]
            [clj-elm.data :as data]
            [clj-elm.util :refer :all]
            [incanter.core :as c :exclude [update]]
            [clojure.core.match :refer [match]]
            [taoensso.timbre.profiling :refer [profile p]])
  (:import [clj_elm.data DataSet]
           [Jama Matrix LUDecomposition])
  (:gen-class))

(def ^:dynamic *sign-reverse* false)

(defn sign
  "The signum function for a real number x."
  ([x]
   {:pre [(number? x)]
    :post [(<= (Math/abs %) 1)]}
   (cond
     (> x 0) 1
     (= x 0) 0
     :else -1)))

(defn reverse-sign
  ([x]
   {:pre [(number? x)]
    :post [(<= (Math/abs %) 1)]}
   (- (sign x))))

(defn make-weights
  "Make a d-dimension-random-feature vector. d is number of dimention of
   feature. All elements are in [-1,1]"
  ([d]
   {:pre [(integer? d)]
    :post [(coll? %)]}
   (take d (repeatedly #(dec (rand 2))))))

(defn make-ass
  ([d L]
   {:pre [(integer? d) (integer? L)]
    :post [(coll? %) (coll? (first %)) (float? (ffirst %))]}
   (take L (repeatedly #(make-weights d)))))

(defn make-bs
  ([L]
   {:pre [(integer? L)]
    :post [(coll? %) (float? (first %))]}
   (take L (repeatedly #(first (make-weights 1))))))

(defn standard-sigmoid
  ([x]
   {:pre [(number? x)]
    :post [(float? %)]}
   (/ 1 (+ 1 (Math/exp (- x))))))

(defn g
  ([x]
   (standard-sigmoid x)))

(defn a-hidden-layer-output
  "Return output of hidden-layer_i with xs. As_i is d-dimension. B_i is number.
   Xs is d-dimension."
  ([as_i b_i xs]
   {:pre [(coll? as_i) (number? b_i) (coll? xs)]
    :post [(float? %)]}
   (p :a-hidden-layer-output
      (g (+ (c/inner-product as_i xs)
            b_i)))))

(defn hidden-layer-output-matrix
  "Ass is d-L-dimension. Xss is d-L-dimesion. Bs is L-dimension."
  ([ass bs xss]
   {:pre [(coll? ass) (coll? (first ass)) (number? (first (first ass)))
          (coll? bs) (number? (first bs))
          (coll? xss) (coll? (first xss))]
    :post [(= (count (first %)) (count ass)) (= (count %) (count xss))]}
   (pmap (fn [xs_i] (pmap #(a-hidden-layer-output %1 %2 xs_i) ass bs)) xss)))

(defn make-double-jarray
  ([array2d]
   {:pre [(coll? array2d) (coll? (first array2d))]}
   (let [newmat (Matrix. (make-array Double/TYPE (count array2d) (count (first array2d))))]
     ;; (dorun (print (format "height %d, width %d\n" (.getRowDimension newmat) (.getColumnDimension newmat))))  ;; for debug
     (dorun
      (for [x (range (.getRowDimension newmat))
            y (range (.getColumnDimension newmat))]
        (.set newmat x y (nth (nth array2d x) y))))
     newmat)))

(defn pseudo-inverse-matrix
  ([mat]
   {:pre [(coll? mat) (coll? (first mat))]
    :post [(c/matrix? %)]}
   (p :pseudo-inverse-matrix
      (let [matrix (c/matrix mat)
            transmat (c/trans matrix)
            n (c/nrow matrix)
            p (c/ncol matrix)]
        (cond
          (= n p) (c/solve matrix)
          (> n p) (-> transmat
                      (c/mmult matrix)
                      (c/solve)
                      (c/mmult transmat))
          (< n p) (-> matrix
                      (c/mmult transmat)
                      (c/solve)
                      (c/mmult matrix)))))))

(defrecord Model [ass bs betas])

(defn train-model
  ([dataset L & {:keys [norm] :or {norm false}}]
   {:pre [(instance? DataSet dataset) (integer? L)]
    :post [(instance? Model %)]}
   (let [normf (if norm data/normalize identity)
         d (data/num-of-feature dataset)
         ass (make-ass d L)
         bs (make-bs L)
         xss (normf (:features dataset))
         H (hidden-layer-output-matrix ass bs xss)
         T (:classes dataset)
         betas (c/to-vect (c/mmult (pseudo-inverse-matrix H) T))]
     (Model. ass bs betas))))

(defn predict
  ([model xs]
   {:pre [(instance? Model model) (coll? xs)]
    :post [(integer? %)]}
   (predict (:ass model) (:bs model) (:betas model) xs))
  ([ass bs betas xs]
   {:pre [(coll? ass) (coll? (first ass)) (coll? bs) (coll? betas) (coll? xs)]}
   (let [signf (if *sign-reverse*
                 reverse-sign
                 sign)]
     (-> (pmap #(* %1 (a-hidden-layer-output %2 %3 xs)) betas ass bs)
         (c/sum)
         (signf)))))

(defn update-exp [fn key exp]
  {:pre [(fn? fn) (keyword? key) (map? exp)]
   :post [(map? %)]}
  (p :update-exp
     (assoc exp key (fn (key exp)))))

(defn count-rate [pred fact exp]
  {:pre [(= (Math/abs pred) (Math/abs fact) 1) (map? exp)]
   :post [(map? %)]}
  (p :count-rate
     (match [pred fact]
       [ 1  1] (update-exp inc :TP exp)
       [ 1 -1] (update-exp inc :FP exp)
       [-1 -1] (update-exp inc :TN exp)
       [-1  1] (update-exp inc :FN exp))))

(defn exp-result [exp]
  {:pre [(map? exp)]
   :post [(map? %)]}
  (p :exp-result
     (let [TP (:TP exp) FP (:FP exp)
           TN (:TN exp) FN (:FN exp)]
       (-> exp
           (assoc :Accuracy (/ (+ TP TN) (+ TP FP TN FN)))
           (assoc :Recall (/ TP (+ TP FN)))
           (assoc :Precision (/ TP (+ TP FP)))))))

(defn _confusion-matrix [preds facts exp]
  (if (or (empty? preds) (empty? facts))
    (exp-result exp)
    (let [pred (first preds)
          fact (first facts)]
      (recur (rest preds) (rest facts) (count-rate pred fact exp)))))

(defn confusion-matrix [preds facts exp]
  {:pre [(coll? preds) (coll? facts) (map? exp)]
   :post [(map? %)]}
  (_confusion-matrix preds facts exp))

(defn evaluation
  ([results facts exp]
   {:pre [(coll? results) (coll? facts) (map? exp)]
    :post [(map? %)]}
   (let [numd (count results)]
     (p :confusion-matrix
        (confusion-matrix results facts exp)))))

(defn select-count
  ([cond coll]
   {:pre [(fn? cond) (coll? coll)]
    :post [(integer? %)]}
   (p :select-count
      (count (filter cond coll)))))

(defn train-to-eval [L train-data test-data]
  (let [exp {:length-train-negative (select-count #(= % -1) (:classes train-data))
             :length-test-negative (select-count #(= % -1) (:classes test-data))
             :length-train-positive (select-count #(= % 1) (:classes train-data))
             :length-test-positive (select-count #(= % 1) (:classes test-data))
             :Accuracy 0.0 :Recall 0.0 :Precision 0.0 :TP 0 :FP 0 :TN 0 :FN 0 :L L}]
    (-> (train-model train-data L)
        (#(pmap (fn [feature] (predict % feature)) (:features test-data)))
        (#(evaluation (:classes test-data) % exp)))))

(defn _cross-validate
  ([dataset a b L]
   {:pre [(instance? DataSet dataset) (integer? a) (integer? b) (> b a) (integer? L)]}
   (let [test (DataSet. (data/extract-list (:classes dataset) a b)
                        (data/extract-list (:features dataset) a b))
         train (DataSet. (data/remove-list (:classes dataset) a b)
                         (data/remove-list (:features dataset) a b))]
     (train-to-eval L train test))))

(defn _print-exp-data [exp]
  {:pre [(map? exp)]
   :post [(string? %)]}
  (str "L:" (:L exp) "\n"
       " length(train_negative): " (:length-train-negative exp) "\n"
       " length(test_negative): " (:length-test-negative exp) "\n"
       " length(train_positive): " (:length-train-positive exp) "\n"
       " length(test_positive): " (:length-test-positive exp) "\n"
       " TP: " (:TP exp) ", FP: " (:FP exp) ", TN: " (:TN exp) ", FN: " (:FN exp) "\n"
       " Accuracy: " (:Accuracy exp)
       ",Recall: " (:Recall exp)
       ",Precision: " (:Precision exp) "\n\n"))

(defn print-exp-data [exp]
  {:pre [(map? exp)]
   :post [(nil? %)]}
  (print (_print-exp-data exp)))

(defn +-exp [exp1 exp2]
  {:pre [(map? exp1) (map? exp2)]
   :post [(map? %)]}
  (p :+-exp
   {:L (:L exp1)
    :length-train-negative (+ (:length-train-negative exp1) (:length-train-negative exp2))
    :length-test-negative (+ (:length-test-negative exp1) (:length-test-negative exp2))
    :length-train-positive (+ (:length-train-positive exp1) (:length-train-positive exp2))
    :length-test-positive (+ (:length-test-positive exp1) (:length-test-positive exp2))
    :TP (+ (:TP exp1) (:TP exp2)) :FP (+ (:FP exp1) (:FP exp2))
    :TN (+ (:TN exp1) (:TN exp2)) :FN (+ (:FN exp1) (:FN exp2))}))

(defn cross-validate
  ([dataset L k]
   {:pre [(instance? DataSet dataset) (integer? k) (integer? L)]
    :post [(map? %)]}
   (let [norm-dataset (DataSet. (:classes dataset) (data/normalize (:features dataset)))
         numd (count (:classes norm-dataset)) ; number of data
         groupn (quot numd k)]                ; number of one group's element
     (->> (take k (iterate #(+ % groupn) 0))
          (pmap #(let [eva (_cross-validate norm-dataset % (+ % (dec groupn)) L)]
                   (dorun [(print-exp-data eva) (flush)])
                   eva))
          (reduce +-exp)
          (exp-result)))))

(defn model-check
  "At first, make model from dataset as a train-dataset, then it validate accuracy of model using dataset as a test-dataset."
  ([dataset L]
   {:pre [(instance? DataSet dataset) (integer? L)]
    :post [(map? %)]}
   (p :model-check
      (let [norm-dataset (DataSet. (:classes dataset) (data/normalize (:features dataset)))]
        (train-to-eval L norm-dataset norm-dataset)))))

(defn -main [& args]
  {:pre [(string? (first args))]}
  (p :main
     (if (= (count args) 9)
       (let [[datap1 classidx1 header1 datap2 classidx2 header2 L n-of-cv sign-m] args]
         (binding [*sign-reverse* (read-string sign-m)]
           (printfl "Start exp\n")
           (def dataset (atom (read-dataset datap1 (read-string classidx1) :header (read-string header1))))
           (printfl (str "Fin read-data " datap1 "\n"))
           (def datasets (atom (read-dataset datap2 (read-string classidx2) :header (read-string header2))))
           (printfl (str "Fin read-data " datap2 "\n"))
           (reset! dataset (data/concat-dataset @dataset @datasets))
           (printfl "Fin data concat\n")
           (reset! dataset (data/shuffle-dataset @dataset))
           (printfl "Fin data shuffle\n")
           (let [result (cross-validate @dataset (read-string L) (read-string n-of-cv))]
             (printfl "Final result:\n")
             (print-exp-data result)
             (flush))
           (System/exit 0))))))
