(ns clj-elm.util)

(defn printfl [obj]
  {:post [(coll? %) (nil? (first %))]}
  [(print obj) (flush)])

(defn reftype? [obj]
  (or (instance? clojure.lang.Ref obj)
      (instance? clojure.lang.Atom obj)))
