(ns clj-elm.data
  (:require [clojure.repl :refer [doc]]
            [incanter.datasets :as da]
            [incanter.core :as c :exclude [update]]
            [incanter.stats :as st]
            [incanter.io :as io]
            [svm.core :as svm]))

(defrecord DataSet [classes features])

(defn parse-lib-svm-data
  ([line]
   {:pre [(integer? (first line)) (map? (last line))]}
   (let [[class featmap] line
         fnum (count featmap)
         features (map #(get featmap %) (range 1 (inc fnum)))]
     [class features])))

(defn data-set
  "Make DataSet from incanter-form or csv dataset. Cidx is column number of class."
  ([dataset cidx]
   {:pre [(c/dataset? dataset)]}
   (DataSet. (pmap int (c/to-vect (c/$ :all cidx dataset))) (c/to-vect (c/$ [:not cidx] dataset)))))

(defn read-dataset
  ([path cidx & {:keys [header] :or {header false}}]
   {:pre [(string? path) (integer? cidx)]}
   (-> (io/read-dataset path :delim \, :header header)
       (data-set cidx))))

(defn read-dataset-lib-svm
  ([path]
   {:pre [(string? path)]}
   (->> (svm/read-dataset path)
        (pmap parse-lib-svm-data)
        (#(DataSet. (map first %) (map second %))))))

(defn concat-dataset
  ([dsa dsb]
   {:pre [(instance? DataSet dsa) (instance? DataSet dsb)]}
   (DataSet. (concat (:classes dsa) (:classes dsb))
             (concat (:features dsa) (:features dsb)))))

(defn shuffle-dataset
  ([dataset]
   {:pre [(instance? DataSet dataset)]}
   (-> (pmap vector (:classes dataset) (:features dataset))
       (shuffle)
       (#(DataSet. (pmap first %) (pmap second %))))))

(defn num-of-feature
  ([dataset]
   {:pre [(coll? dataset)]}
   (count (first (:features dataset)))))

(defn ith-feature-list
  ([dataset i]
   {:pre [(coll? dataset) (integer? i)]}
   (c/$ i (c/to-dataset dataset))))

(defn each-ith-f
  ([f dataset]
   (->> (range 0 (count (first dataset)))
        (pmap #(f (ith-feature-list dataset %)))
        (vec))))

(defn each-ith-mean
  ([dataset]
   {:pre [(coll? dataset)]}
   (each-ith-f st/mean dataset)))

(defn each-ith-sd
  ([dataset]
   {:pre [(coll? dataset)]}
   (each-ith-f st/sd dataset)))

(defn normalize
  ([dataset]
   {:pre [(coll? dataset)]}
   (let [means (each-ith-mean dataset)
         sds (each-ith-sd dataset)]
     (->> (c/to-vect dataset)
          (pmap #(map - % means))
          (pmap #(map / % sds))))))

(defn remove-list
  ([lst a b]
   {:pre [(coll? lst) (integer? a) (integer? b) (> b a)]}
   (concat (take a lst) (drop (inc b) lst))))

(defn extract-list
  ([lst a b]
   {:pre [(coll? lst) (integer? a) (integer? b) (> b a)]}
   (take (inc (- b a)) (drop a lst))))
