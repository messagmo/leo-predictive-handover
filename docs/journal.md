# Journal du projet

## Étape 1 : environnement
- Colab, dépôt GitHub, bibliothèques installées (sgp4, skyfield, gymnasium).
- 10 611 satellites Starlink téléchargés depuis CelesTrak.

## Étape 2 : orbit.py
- Filtre du HHS (14,9 à 15,2 tours/jour, |B*| ≤ 1e-4) : 57 satellites gardés sur 10 611, trop peu.
- Plage élargie à 14,9 à 16,0 tours/jour : 10 576 satellites gardés.
- Visibilité à 5° d'élévation depuis Riyad : 226,4 satellites visibles en moyenne (min 209, max 248).
- Validation croisée avec Skyfield : écart d'élévation de 0,0006° sur un satellite visible.
