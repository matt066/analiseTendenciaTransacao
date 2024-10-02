package com.m.analiseTendenciaTransacao;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;
import java.io.File;
import java.io.IOException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Service
public class ModelService {

    private static final Logger log = LoggerFactory.getLogger(ModelService.class);

    private MultiLayerNetwork model;

    @PostConstruct
    public void init() {
        try {
            File modelFile = new File("model.zip");
            if (modelFile.exists()) {
                log.info("Carregando o modelo a partir de {}", modelFile.getPath());
                this.model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
                log.info("Modelo carregado com sucesso.");
            } else {
                log.warn("Arquivo de modelo não encontrado em {}", modelFile.getPath());
            }
        } catch (IOException e) {
            log.error("Erro ao carregar o modelo: ", e);
        }
    }

    public MultiLayerNetwork getModel() {
        return model;
    }
}