/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.assignment1;

import java.io.Serializable;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.sql.AnalysisException;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 *
 * @author Sayed
 */


public class kmean   {
     

    public static void  main (String []args) throws AnalysisException{
    
        SparkSession newsession;
          newsession = SparkSession.builder()
                  .appName("kmean")
                  .master("local[*]")
                  .config("spark.sql.warehouse.dir", "file:///D://")
                  .getOrCreate();
    
        Dataset<Row> data = newsession
                .read().option("header", "true")
                .csv("C://Users//Sayed//Documents//NetBeansProjects//hack_data.csv");
        data.show();
          JavaRDD<kmeandatasetclass> data1 = data.toJavaRDD().map((Row t1) -> {
              // throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
              kmeandatasetclass cr = new kmeandatasetclass();
              cr.setSession_Connection_Time(t1.getString(0));
              cr.setBytes_Transferred(t1.getString(1));
              cr.setKali_Trace_Used(t1.getString(2));
              cr.setServers_Corrupted(t1.getString(3));
              cr.setPages_Corrupted(t1.getString(4));
              cr.setLocation(t1.getString(5));
              cr.setWPM_Typing_Speed(t1.getString(6));
              
              return cr;
        });
           // Dataset<Row> data2 = session.createDataFrame(data1, cancerobservation.class);
        Dataset<Row> data2 = newsession.createDataFrame(data1, kmeandatasetclass.class);
          //data2.write().csv("E:\\withcoooo");
          data2.printSchema();
          StringIndexerModel index1 = new StringIndexer().setInputCol("session_Connection_Time")
                .setOutputCol("Osession_Connection_Time").fit(data2);
           
            StringIndexerModel index2 = new StringIndexer().setInputCol("servers_Corrupted")
                    .setOutputCol("Oservers_Corrupted").fit(data2);
             StringIndexerModel index3 = new StringIndexer().setInputCol("pages_Corrupted")
                    .setOutputCol("Opages_Corrupted").fit(data2);
              StringIndexerModel index4= new StringIndexer().setInputCol("location")
                    .setOutputCol("Olocation").fit(data2);
              StringIndexerModel index5 = new StringIndexer().setInputCol("kali_Trace_Used")
                    .setOutputCol("Okali_trace_used").fit(data2);
           
              StringIndexerModel index6= new StringIndexer().setInputCol("bytes_Transferred")
                    .setOutputCol("ObytesTransferred").fit(data2);
          StringIndexerModel index7 = new StringIndexer().setInputCol("WPM_Typing_Speed")
                    .setOutputCol("OWPM_Typing_Speed").fit(data2);
          
           
          Pipeline pipeline1 = new Pipeline()
                .setStages(new PipelineStage[]{index1, index2, index3, index4, index5, index6, index7});
          Dataset <Row> data3= pipeline1.fit(data2).transform(data2);
          System.out.println("loooooooooooooooooooooooooooooooool");
          data2.show();
            VectorAssembler assemblar = new VectorAssembler()
                  .setInputCols(new String[]{"Osession_Connection_Time","ObytesTransferred","Okali_trace_used","Oservers_Corrupted","Opages_Corrupted","Olocation","OWPM_Typing_Speed"})
                  .setOutputCol("features");
             Dataset<Row> featuredata = assemblar.transform(data3);
        
        VectorIndexerModel featureIndexer = new VectorIndexer()
                .setInputCol("features")
                .setOutputCol("indexerfeatures")
                .setMaxCategories(4) // features with > 4 distinct values are treated as continuous.
                .fit(featuredata);
         KMeans kmean = new KMeans();//.setPredictionCol("oprediction");
        kmean.setK(3);
        kmean.setFeaturesCol("indexerfeatures");
        kmean.setSeed(1L);
        Pipeline p2 = new Pipeline()
                .setStages(new PipelineStage[]{featureIndexer,kmean});
        Dataset <Row> fe= p2.fit(featuredata).transform(featuredata);
        fe.show();
        
        Dataset <Row> []splits = featuredata.randomSplit(new double []{0.8,0.2} );
        Dataset <Row> trainingData = splits[0];
        Dataset <Row> testdata = splits[1];
       
        
        
        PipelineModel pm = p2.fit(trainingData);
        Dataset<Row> predect = pm.transform(testdata);
        predect.show();
        
        
//        
          /*Dataset<Row> vectorized_df= assemblar.transform(data1);
        
        ///////////////////////////////////.
        KMeans kmeans = new KMeans().setK(3).setSeed(1L);
        KMeansModel model = kmeans.fit(vectorized_df);
       
        
        Dataset<Row> dataset = model.transform(data1);
        dataset.show();
    */
    }
}
