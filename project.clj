(defproject clj-elm "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :plugins [[cider/cider-nrepl "0.10.0-SNAPSHOT"]
            [lein-midje "3.1.3"]
            [lein-cloverage "1.0.2"]]
  :repl {:dependencies [[org.clojure/tools.nrepl "0.2.12"]]}
  :dependencies [[org.clojure/clojure "1.7.0"]
                 [org.clojure/core.match "0.3.0-alpha4"]
                 [incanter "1.5.6"]
                 [midje "1.7.0-beta1"]
                 [svm-clj "0.1.3"]
                 [criterium "0.4.3"]
                 [com.taoensso/timbre "4.1.4"]]
  :global-vars {*assert* true}
  :main clj-elm.core)
