# Dataset sintético realista de arquitetura

Contém 60 diagramas de arquitetura em estilo visual realista, com labels em formato YOLO.

## Classes
- 0: API Gateway
- 1: Database
- 2: Cache
- 3: Message Queue
- 4: User
- 5: Application Server
- 6: Load Balancer

## Estrutura
- images/: imagens PNG
- labels/: anotações YOLO (.txt)
- data.yaml: arquivo de configuração

## Observações
- Este dataset foi gerado programaticamente para acelerar fine-tuning e validação visual.
- As imagens são sintéticas, mas com composição inspirada em diagramas reais de cloud/system architecture.
- As labels cobrem apenas as 7 classes solicitadas.