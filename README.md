# Bakalářská práce — LaTeX projekt

## Struktura

```
main.tex              hlavní soubor, preambule, vstupy kapitol
refs.bib              literatura
chapters/             jednotlivé kapitoly
figures/              obrázky (PDF pro sazbu, zdroje vedle nich)
code/                 skripty, které ty obrázky generují
code/hardware/        model soupravy pro Gazebo, tištěné díly
.vscode/settings.json nastavení editoru
```

## Jak psát

Ve `main.tex` jsou vstupy kapitol zakomentované. **Odkomentuj vždy jen tu,
kterou zrovna píšeš** — nedokončené kapitoly pak nebrání v překladu.

Uložením souboru se spustí překlad (LaTeX Workshop). PDF se otevře ve vedlejším
panelu, cmd+klik v něm skočí na odpovídající řádek ve zdroji.

## Obrázky

`figures/` obsahuje hotová PDF. Skripty, které je generují, jsou v `code/`.
Po doplnění změřených rozměrů stačí skripty pustit znovu a obrázky se
přegenerují — proto jsou v repozitáři obojí.

| obrázek | zdroj |
|---|---|
| `geometry.pdf` | `figures/geometry.tex` |
| `fig1`–`fig5` | `code/feasibility_study.py` |
| `sim1`–`sim4` | `code/simulation4.py` |

## Před odevzdáním

Projeď log a najdi všechna zbylá `\TODO` — každé vypisuje varování.

## Poznámky a analýzy

Pracovní poznámky, plán kapitol a rozbory jsou mimo repozitář,
v `AIOS/Bachelor thesis/`.
