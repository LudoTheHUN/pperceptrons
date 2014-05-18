(defproject pperceptrons "0.1.0-SNAPSHOT"
  :description "A Clojure library designed to implement parallel perceptions"
  :url "https://github.com/LudoTheHUN/pperceptrons"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.6.0"]
                 [net.mikera/core.matrix "0.22.0"]
                 [net.mikera/vectorz-clj "0.22.0"]]
  :profiles {:dev {:dependencies [[criterium "0.4.3"]]}})


