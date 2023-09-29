# <span style="color:cyan"> Predict Health Outcomes of Horses
### <span style="color:lightblue"> DATASET obtenido desde las competiciones de kaggle.com

#### <span style="color:#87CEEB"> Objetivo: Predecir el estado de salud de los caballos en estudio.

#### <span style="color:#87CEEB"> La métrica que se busca mejorar es el micro-averaged F1-Score. Este valor se calcula usando la totalidad de verdaderos positivos, falsos positivos y falsos negativos, en lugar de calcular el f1 score individualmente para cada clase.

https://www.kaggle.com/competitions/playground-series-s3e22/overview/evaluation

### <span style="color:orange"> Conjunto de datos:
    - Surgery: Identificador de pasajero.
        1 = Si, tuvo cirugía
        2 = Fue tratado sin cirugía

    - Age: Edad
        1 = Caballo adulto
        2 = Joven (< 6 meses)
    
    - Hospital Number:
        - identificador numérico
        - el número asignado al caballo puede no ser único si se ha tratado más de una vez

    - Rectal temperature: temperatura rectal
        - lineal
        - en grados celcius
        - una temperatura elevada puede ocurrir debido a infección
        - la temperatura puede disminuir cuando el animal está en shock tardío
        - la temperatura normal es 37.8℃
        - este parámetro usualmente cambia a medida que el problema progresa, por ejemplo, puede comenzar normal, luego aumenta debido a la lesión, posteriormente vuelve al rango normal a medida que el caballo entra en shock

    - Pulse: pulso
        - lineal
        - ritmo cardíaco en latidos por minuto
        - es un reflejo de la condición del corazón. 30-40 latidos es normal para adultos
        - es raro tener un ritmo más bajo de lo normal, aunque los caballos atléticos pueden tener un ritmo de 20-25 lpm
        - los animales con lesiones dolorosas o que sufren shock circulatorio pueden tener una frecuencia cardíaca elevada

    - Respiratory rate: frecuencia respiratoria
        - lineal
        - la frecuencia normal es 8 a 10
        - la utilidad es dudosa debido a las grandes fluctuaciones

    - Temperature of extremities: temperatura de las extremidades. Una indicación subjetiva de la circulación periférica
        Posibles valores:
            1 = Normal
            2 = Warm
            3 = Cool
            4 = Cold
        . cool a cold indican posible shock
        . extremidades calientes se deberían correlacionar con una elevada temperatura rectal
    
    - Peripheral pulse: pulso periférico. Subjetivo
        Posibles valores:
            1 = normal
            2 = increased
            3 = reduced
            4 = absent
        . normal o increased son indicativos de circulación adecuada, mientras que reduced o absent indican mala perfusión

    - Mucous membranes: membranas mucosas. una subjetiva medición de color
        Posibles valores:
            1 = normal pink
            2 = bright pink
            3 = pale pink
            4 = pale cyanotic
            5 = bright red / injected
            6 = dark cyanotic
        . normal pink o bright pink indican un normal o leve aumento en la circulación
        . pale pink puede ocurrir en un shock temprano
        . pale cyanotic y dark cyanotic son indicativos de un serio compromiso circulatorio
        . bright red / injected es más indicativo de septicemia

    - Capillary refill time: tiempo de llenado capilar. Un critero clínico. Mientras más demore, más bajos serán los valores
        Posibles valores:
            - 1: < 3 segundos
            - 2: >= 3 segundos

    - Pain: Dolor. criterio subjetivo del nivel de dolor del caballo
        Posibles valores:
            1: None
            2: alert
            3: depressed
            4: moderate
            5: mild_pain
            6: severe_pain
            7: extreme_pain
        . NO debe tratarse como una variable ordenada o discreta.
        . En general, cuanto más doloroso, más probable es que requiera cirugía. El tratamiento previo del dolor puede enmascarar el nivel del dolor hasta cierto punto.

    - Peristalsis. Una indicación de la actividad en el intestino del caballo. A medida que el intestino se distiende más o el caballo se vuelve más tóxico, la actividad disminuye
        Posibles valores:
        1 = hypermotile
        2 = normal
        3 = hypomotile
        4 = absent
    
    - Abdominal distension: Distensión abdominal
        Posibles valores:
            1 = none
            2 = slight
            3 = moderate
            4 = severe
        . Es probable que un animal con distensión abdominal sienta dolor y tenga una motilidad intestinal reducida.
        . Es probable que un caballo con distensión abdominal severa requiera cirugía solo para aliviar la presión.

    - Nasogastric tube: Tubo nasogástrico. Se refiere a cualquier gas proveniente del tubo
        Posibles valores:
            1 = none
            2 = slight
            3 = significant
        . Es probable de una gran cantidad de gas le genere molestias al caballo

    - Nasogastric reflux: reflujo nasogástrico
        Posibles valores:
            1 = none
            2 = > 1 litro
            3 = < 1 litro
        . Cuanto mayor sea la cantidad de reflujo, mayor será la probabilidad de que exista alguna obstrucción grave en el paso del líquido desde el resto del intestino

    - Nasogastric reflux PH: Acidez del reflujo nasogástrico
        . lineal
        . La escala es de 0 a 14, siendo 7 neutral.
        . Los valores normales están en el rango de 3 a 4.

    - Rectal examination - feces: Examinación rectal - fecas
        Posibles valores:
        1 = normal
        2 = increased
        3 = decreased
        4 = absent
        . absent posiblemente indica una obstrucción
    
    - Abdomen
        Posibles valores:
            1 = normal
            2 = other
            3 = firm feces in the large intestine
            4 = distended small intestine
            5 = distended large intestine
        . 3 es probablemente una obstrucción causada por un impacto mecánico y normalmente se trata médicamente
        . 4 y 5 indican una lesión quirúrgica
    
    - Packed cell volume
        . lineal
        . el número de glóbulos rojos por volumen en la sangre
        . el rango normal es de 30 a 50. El nivel aumenta a medida que la circulación se ve comprometida o cuando el animal se deshidrata.

    - Total protein: total de proteínas
        . lineal
        . los valores normales se encuentran en el rango de 6 a 7,5 (g/dL)
        . cuanto mayor sea el valor mayor será la deshidratación

    - Abdominocentesis appearance: apariencia de abdominocentesis. Se introduce una aguja en el abdomen del caballo y se obtiene líquido de la cavidad abdominal
        Posibles valores:
            1 = clear
            2 = cloudy
            3 = serosanguinous
        . El líquido normal es claro, mientras que el turbio o serosanguinolento indica un intestino comprometido.
    
    - Abdomcentesis total protein
        . lineal
        . cuanto mayor sea el nivel de proteína, más probabilidades habrá de tener un intestino comprometido. Los valores están en g/dL.

    - Surgical lesion: Lesión quirúrgica
        . retrospectivamente, ¿el problema (lesión) fue quirúrgico?. Todos los casos son operados o autopsiados para que siempre se conozca este valor y el tipo de lesión
        Posibles valores:
            1 = Yes
            2 = No

    - lesion_1, lesion_2, lesion_3: Tipo de lesión
        Primer dígito indica el sitio de la lesión:
            1 = gastric
            2 = sm intestine
            3 = lg colon
            4 = lg colon and cecum
            5 = cecum
            6 = transverse colon
            7 = retum/descending colon
            8 = uterus
            9 = bladder
            11 = all intestinal sites
            00 = none

        Segundo dígito es el tipo de lesión:
            1 = simple
            2 = strangulation
            3 = inflammation
            4 = other

        Tercer dígito es el subtipo:
            1 = mechanical
            2 = paralytic
            0 = n/a
        
        Cuarto dígito es el código específico
            1 = obturation
            2 = intrinsic
            3 = extrinsic
            4 = adynamic
            5 = volvulus/torsion
            6 = intussuption
            7 = thromboembolic
            8 = hernia
            9 = lipoma/slenic incarceration
            10 = displacement
            0 = n/a

    - cp_data: ¿Hay datos de patología presentes para este caso?
        Posibles valores:
            1 = Yes
            2 = No
        . Esta variable no tiene importancia ya que los datos de patología no se incluyen ni se recopilan para estos casos

    - Outcome: Resultado. ¿Qué pasó finalmente con el caballo?
        Posibles valores:
            1 = lived
            2 = died
            3 = was euthanized

### <span style="color:orange"> Herramientas Utilizadas:
    - Python 3.9.17
    - Bibliotecas de análisis de datos: Pandas, NumPy.
    - Bibliotecas de visualización: Matplotlib, Seaborn.
    - Biblioteca de aprendizaje automático: scikit-learn.

### <span style="color:orange"> Resultados y Conclusiones:
    
