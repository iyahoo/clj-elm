(ns clj-elm.data
  (:require [clojure.repl :refer [doc]]
            [incanter.datasets :as da]
            [incanter.core :as c]
            [incanter.stats :as st]
            [incanter.io :as io]))

(def australian
  (io/read-dataset "data/australian.csv" :delim \, :header true))

(defn num-of-feature 
  ([dataset]
   {:pre [(coll? dataset)]}
   (c/ncol (c/to-dataset dataset))))

(defn num-of-data 
  ([dataset]
   {:pre [(or (c/matrix? dataset) (c/dataset? dataset))]}
   (c/nrow dataset)))

(defrecord DataSet [classes features])

(defn data-set
  "Make DataSet from incanter-form dataset. Cidx is row number of class."
  ([^incanter.core.Dataset dataset cidx]
   {:pre [(c/dataset? dataset)]}
   (DataSet. (map int (c/to-vect (c/$ :all cidx dataset))) (c/to-vect (c/$ [:not cidx] dataset)))))

(defn get-features 
  ([line]
   {:pre [(coll? line)]}
   (butlast line)))

(defn class-label 
  ([dataset]
   {:pre [(or (c/matrix? dataset) (c/dataset? dataset))]}
   (map int (c/to-vect (c/$ :all 14 dataset)))))

(defn ith-feature-list 
  ([dataset i]
   {:pre [(coll? dataset)]}
   (c/$ i (c/to-dataset dataset))))

(defn each-ith-f 
  ([f dataset]
   {:pre [(coll? dataset)]}
   (->> (range 0 (count (first dataset)))
        (map #(f (ith-feature-list dataset %)))
        (vec))))

(defn each-ith-mean 
  ([dataset]
   (each-ith-f st/mean dataset)))

(defn each-ith-sd 
  ([dataset]
   (each-ith-f st/sd dataset)))

(defn normalize 
  ([dataset]
   {:pre [(coll? dataset)]}
   (let [means (each-ith-mean dataset)
         sds (each-ith-sd dataset)]    
     (->> (c/to-vect dataset)
          (map #(map - % means))
          (map #(map / % sds))
          (c/to-dataset)))))
