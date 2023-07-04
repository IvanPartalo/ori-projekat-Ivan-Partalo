# ori-projekat-Ivan-Partalo

## Simulacija robota u skladištu ##

U rešenju se nalazi nekoliko fajlova, od kojih neki služe za treniranje
neuronske mreže, a neki za testiranje trenirane mreže (modela).
Osim toga tu se nalazi i nekoliko vec treniranih modela, koji se mogu
učitati i istestirati.

Da bi se pokrenulo rešenje potrebno je pokrenuti neku od skripti sa
nazivom evaluacija u sebi:

-   oriProjekatEvaluacija (robot koji prebacuje sve kutije do ciljne površine na sredini mape)
-   oriProjekatEvaluacijaEnemiesNovi (robot koji prebacuje samo određene kutije do ciljne površine na vrhu mape)

Za korišćenje je potrebno:

-   python minimalna verzija 3.6
-   torch (korišćena verzija 2.0.1)
-   pygame (korišćena verzija 2.3.0)

Napomena: ako se stvara problem zbog putanje virtual environment-a, može se pokrenuti preko komandne linije:
python oriProjekatEvaluacija.py
