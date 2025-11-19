## Objetivo
- Remover o bloco try/except duplicado após `if __name__ == '__main__': main()` em `main.py` que está fora de qualquer função e causa erro de sintaxe/IDE.

## Mudança
- Apagar exatamente as linhas `main.py:866–879` (bloco que treina o classificador), pois essa lógica já está dentro de `_train_and_evaluate`.

## Validação
- Rodar `pytest -q` para garantir que não há regressões.
- (Opcional) Executar `main()` para verificar que o fluxo roda e o classificador é salvo quando houver rótulos.

## Entregáveis
- Arquivo `main.py` sem o bloco solto.
- Confirmação dos testes passando.