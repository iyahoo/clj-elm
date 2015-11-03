(ns clj-elm.core
  (:require [clj-elm.data :refer [read-dataset normalize]]
            [clj-elm.data :as data]
            [clj-elm.util :refer :all]
            [incanter.core :as c :exclude [update]]
            [clojure.core.match :refer [match]])
  (:import [clj_elm.data DataSet])
  (:gen-class))

(defn sign
  "The signum function for a real number x."
  ([x]
   {:pre [(number? x)]
    :post [(< (Math/abs %) 2)]}
   (cond
     (> x 0) 1
     (= x 0) 0
     :else -1)))

(defn make-weights
  "Make a d-dimension-random-feature vector. d is number of dimention of
   feature. All elements are in [-1,1]"
  ([d]
   {:pre [(integer? d)]}
   (take d (repeatedly #(dec (rand 2))))))

(defn make-ass
  ([d L]
   {:pre [(integer? d) (integer? L)]}
   (take L (repeatedly #(make-weights d)))))

(defn make-bs
  ([L]
   {:pre (integer? L)}
   (take L (repeatedly #(first (make-weights 1))))))

(defn standard-sigmoid
  ([x]
   {:pre [(number? x)]
    :post [(number? %)]}
   (/ 1 (+ 1 (Math/exp (- x))))))

(defn bipolar-sigmoid
  ([x]
   {:pre [(number? x)]
    :post [(number? %)]}
   (dec (/ 2 (+ 1 (Math/exp (- x)))))))

(defn g
  ([x]
   (standard-sigmoid x)))

(defn a-hidden-layer-output
  "Return output of hidden-layer_i with xs. As_i is d-dimension. B_i is number.
   Xs is d-dimension."
  ([as_i b_i xs]
   {:pre [(coll? as_i) (number? b_i) (coll? xs)]}
   (g (+ (c/inner-product as_i xs)
         b_i))))

(defn hidden-layer-output-matrix
  "Ass is d-L-dimension. Xss is d-L-dimesion. Bs is L-dimension."
  ([ass bs xss]
   {:pre [(coll? ass) (coll? (first ass)) (number? (first (first ass)))
          (coll? bs) (number? (first bs))
          (coll? xss) (coll? (first xss))]
    :post [(= (count (first %)) (count ass)) (= (count %) (count xss))]}
   (pmap (fn [xs_i] (pmap #(a-hidden-layer-output %1 %2 xs_i) ass bs)) xss)))

(defn pseudo-inverse-matrix
  ([mat]
   {:pre [(coll? mat) (coll? (first mat))]}
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
                   (c/mmult matrix))))))

(defrecord Model [ass bs betas])

(defn train-model
  ([dataset L & {:keys [norm] :or {norm false}}]
   {:pre [(instance? DataSet dataset) (integer? L)]}
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
   {:pre [(instance? Model model) (coll? xs)]}
   (predict (:ass model) (:bs model) (:betas model) xs))
  ([ass bs betas xs]
   {:pre [(coll? ass) (coll? (first ass)) (coll? bs) (coll? betas) (coll? xs)]}
   (-> (pmap #(* %1 (a-hidden-layer-output %2 %3 xs)) betas ass bs)
       (c/sum)
       (sign))))

(defn update-exp [fn key exp]
  {:pre [(fn? fn) (keyword? key) (map? @exp)]}
  (reset! exp (assoc @exp key (fn (key @exp))))
  exp)

(defn count-rate [pred fact exp]
  {:pre [(= (Math/abs pred) (Math/abs fact) 1) (map? @exp)]}
  (match [pred fact]
    [ 1  1] (update-exp inc :TP exp)
    [ 1 -1] (update-exp inc :FP exp)
    [-1 -1] (update-exp inc :TN exp)
    [-1  1] (update-exp inc :FN exp)))

(defn confusion-matrix [preds facts exp]
  {:pre [(coll? preds) (coll? facts) (map? @exp)]}
  (if (or (empty? preds) (empty? facts))
    (let [TP (:TP @exp) FP (:FP @exp)
          TN (:TN @exp) FN (:FN @exp)]
      (-> @exp
          (assoc :Accuracy (/ (+ TP TN) (+ TP FP TN FN)))
          (assoc :Recall (/ TP (+ TP FN)))
          (#(reset! exp (assoc % :Precision (/ TP (+ TP FP)))))))
    (let [pred (first preds)
          fact (first facts)]
      (recur (rest preds) (rest facts) (count-rate pred fact exp)))))

(defn evaluation
  ([results facts exp]
   {:pre [(coll? results) (coll? facts) (map? @exp)]}
   (let [numd (count results)]
     (confusion-matrix results facts exp))))

(defn select-count
  ([cond coll]
   {:pre [(fn? cond) (coll? coll)]}
   (count (filter cond coll))))

(defn _cross-validate
  ([dataset a b L]
   {:pre [(instance? DataSet @dataset) (integer? a) (integer? b) (> b a)]}
   (let [test (DataSet. (data/extract-list (:classes @dataset) a b)
                        (data/extract-list (:features @dataset) a b))
         train (DataSet. (data/remove-list (:classes @dataset) a b)
                         (data/remove-list (:features @dataset) a b))
         exp (atom {:num (str "Data " a " to " b)
                    :length-train-negative (select-count #(= % -1) (:classes train))
                    :length-test-negative (select-count #(= % -1) (:classes test))
                    :length-train-positive (select-count #(= % 1) (:classes train))
                    :length-test-positive (select-count #(= % 1) (:classes test))
                    :Accuracy 0.0 :Recall 0.0 :Precision 0.0 :TP 0 :FP 0 :TN 0 :FN 0 :L L})]
     (-> (train-model train L)
         (#(pmap (fn [feature] (predict % feature)) (:features test)))
         (#(evaluation (:classes test) % exp))))))

(defn print-exp-data [expdata]
  (print "L: " (:L expdata) "\n"
         "length(train_negative): " (:length-train-negative expdata) "\n"
         "length(test_negative): " (:length-test-negative expdata) "\n"
         "length(train_positive): " (:length-train-positive expdata) "\n"
         "length(test_positive): " (:length-test-positive expdata) "\n"
         "TP: " (:TP expdata) ", FP: " (:FP expdata) ", TN: " (:TN expdata) ", FN: " (:FN expdata) "\n"
         "Accracy: " (:Accuracy expdata "\n")
         "Recall: " (:Recall expdata "\n")
         "Precision: " (:Precision expdata) "\n\n"))

(defn cross-validate
  ([dataset L k]
   {:pre [(instance? DataSet dataset) (integer? k) (integer? L)]}
   (let [norm-dataset (atom (DataSet. (:classes dataset) (data/normalize (:features dataset))))
         numd (count (:classes @norm-dataset)) ; number of data
         groupn (quot numd k)]                ; number of one group's element
     (->> (take k (iterate #(+ % groupn) 0))
          (pmap #(let [eva (_cross-validate norm-dataset % (+ % (dec groupn)) L)]
                   (dorun [(print-exp-data eva) (flush)])
                   (:Accuracy eva)))
          (reduce +)
          (* (/ 1 k))))))


(defn -main [& args]
  (if (= (count args) 6)
    (do
      (printfl "Start exp\n")
      (def dataset (atom (read-dataset (first args) (read-string (second args)) :header (read-string (nth args 2)))))
      (printfl "Fin read-data outguess_all_0.0.csv\n")
      (def datasets (atom (read-dataset (nth args 3) (read-string (nth args 4)) :header (read-string (nth args 5)))))
      (printfl "Fin read-data outguess_all_1.0.csv\n")
      (reset! dataset (data/concat-dataset @dataset @datasets))
      (printfl "Fin data concat\n")
      (reset! dataset (data/shuffle-dataset @dataset))
      (printfl "Fin data shuffle\n")
      (printfl (str (cross-validate @dataset 100 10) "\n"))
      (System/exit 0))))
