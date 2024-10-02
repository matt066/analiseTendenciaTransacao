package com.m.analiseTendenciaTransacao;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class NeuralNetworkConfig {

    private int numInputs = 3;  // Número de entradas, ajuste conforme sua necessidade
    private int numOutputs = 2;  // Número de saídas, por exemplo, para classificação binária
    private int numHiddenNodes = 20;  // Nós em uma camada oculta
    private int numClasses = 2;  // Número de classes para classificação

    @Bean
    public MultiLayerNetwork multiLayerNetwork() {
        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.01))
                .list()  // Inicia a configuração das camadas
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(numHiddenNodes).nOut(numClasses)
                        .build());

        MultiLayerConfiguration conf = builder.build(); // Constrói a configuração
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        return network;
    }
}
