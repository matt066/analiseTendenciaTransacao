package com.m.analiseTendenciaTransacao;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.context.request.WebRequest;
import org.deeplearning4j.exception.DL4JInvalidInputException;

@ControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(DL4JInvalidInputException.class)
    public ResponseEntity<?> handleDL4JInvalidInputException(DL4JInvalidInputException ex, WebRequest request) {
        String message = "Erro de entrada na rede neural: Verifique se a quantidade de features fornecidas corresponde à esperada pela rede.";
        // Log detalhado para debug
        System.out.println("Erro encontrado: " + ex.getMessage());
        return new ResponseEntity<>(message, HttpStatus.BAD_REQUEST);
    }

    // Outros handlers de exceção conforme necessário
}

