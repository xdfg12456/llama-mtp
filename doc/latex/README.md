# NYCU Chinese Master Thesis Template

* 此專案為「國立陽明交通大學」「中文碩士」論文模板。
* 本人論文已上傳圖書館通過審核，請放心服用。
    > 除了「附錄」以外。請查閱[「免責聲明」](#一些免責聲明)段落。
* 為了方便閱讀和改寫，大部分的註解和說明都已經是寫中文了，可以看一下再動筆。
* 已更新學校提供之範本與相關格式規範（Sep. 16th, 2023）

## Template編譯與輸出方法

### 編譯方法與順序

1. 總結：TeX檔案需使用`xelatex`編譯，Bib的部分則需使用`biber`編譯。
    > * 那個比較潮用`pdflatex`搭配對utf-8支援的使用方法，我怎麼用就是不成功啦，QQ。所以中文是使用`xeCJK`套件完成的，所以需要使用`xelatex`才可以編。
    > * 阿那個Bib好像大部分的人都是用`bibtex`編的，但是他不會自動對URL出現`visited on`字樣，我不是很開心，所以我就改用`biblatex`搭配`biber`編譯了，然後我覺得很棒。
2. 編譯順序：
    1. 方法一：`xelatex` -> `biber` -> `xelatex` -> `xelatex`（我自己的編法）
        ```bash
        xelatex main.tex
        biber main
        xelatex main.tex
        xelatex main.tex
        ```
    2. 方法二：用很潮的`latexmk`就好了（我試過可以用的編法）
        ```bash
        latexmk -xelatex main.tex
        ```
3. 我有準備makefile，所以
    ```bash
    ## 一般Linux的同學：
        #  編譯
        make            

        # 清掉編譯過程產生的垃圾們
        make clean
    
    ## Windows的同學：
        #  編譯
        make -f WinMakefile

        # 清掉編譯過程產生的垃圾們
        make -f WinMakefile clean
    ```
4. 阿如果只是想要看一下剛剛打的一行字會有甚麼效果的話，就只要使用`xelatex main.tex`編一次就可以看到了（link、目錄和編號都有可能會壞掉，但是東西出現的位置和順序基本上不會有差），這樣比較快。

### 輸出格式選項

1. 可選格式
    * 論文階段：初稿/定稿 (**`draft`**/`final`)
        * `draft` - 會幫你在封面加上「初稿」
        * `final` - 就是正式版的東西，nothing special
        * 不管哪一個都會有校徽浮水印，因為我不喜歡「DRAFT」浮水印，很醜。
    * 論文使用模式：印刷/圖書館上傳格式 (**`print`**/`upload`)
        * `print` - 輸出是指印出來的時候會包含的東西，所以會幫你把那一堆審定書、授權書都加上來。
        * `upload` - 顧名思義，是拿來上傳圖書館用的，就不會有那一堆文件。

2. 修改方式
    * 修改 [`main.tex: 14`](main.tex#L14)
    * 將欲輸出的格式填入方框中以逗號隔開，如下例為「論文印刷定稿」。
        ```latex=9
        \documentclass[final, print]{Class/NYCU-Thesis}
        ```

## 關於這份Template

### 一些說明和注意事項

* 請於[`Config/config.tex`](Config/config.tex)設定論文各項參數，各參數皆附有註解說明用途。
* **請一定要去[`Config/fonts.tex`](Config/fonts.tex)設定中文字體**，要不然很有可能無法編譯。
* Template有提供以下幾個額外的Command
    * `\AlgoHRule`
        * 在Pseudo Code環境中，製造一條水平線。
        * 使用情境請參考：[`5-Chapters/3-Design.tex: 30`](5-Chapters/3-Design.tex#L30)。
    * Table Column Type `C`
        * 在`tabular`環境中可以指定格子寬度然後自動換行並且置中。
        * 使用情境請參考：[`5-Chapters/4-Evaluation.tex: 55`](5-Chapters/4-Evaluation.tex#L55)。
    * Table Column Type `Y`
        * 在`tabularx`環境中平均分配格子寬度。
        * 使用情境請參考：[`5-Chapters/4-Evaluation.tex: 79`](5-Chapters/4-Evaluation.tex#L79)。
* 本Template有提供許多LaTex使用範例。
    * 各範例說明：
        * [`5-Chpaters/1-Introduction.tex`](5-Chapters/1-Introduction.tex)：提供切分段落的範例。
        * [`5-Chpaters/2-RelatedWorks.tex`](5-Chapters/2-RelatedWorks.tex)：提供如何`\cite`別人的文獻的範例。
        * [`5-Chpaters/3-Design.tex`](5-Chapters/3-Design.tex)：提供如何在論文中插入公式和pseudo code段落的範例。
        * [`5-Chpaters/4-Evaluation.tex`](5-Chapters/4-Evaluation.tex)：提供如何插入表格和圖片的範例。
        * [`5-Chpaters/5-Conclusion.tex`](5-Chapters/5-Conclusion.tex)：提供如何寫註腳的範例。
        * [`6-Reference/thesis.bib`](6-Reference/thesis.bib)：提供多種文獻的引入範例。
    * 範例文章來源：
        * [亂數假文產生器 Chinese Lorem Ipsum](http://www.richyli.com/tool/loremipsum/)
        * [唬爛產生器](https://howtobullshit.me/)
        * [Drpapaer 期末報告產生器](https://drpaper.neocities.org/)
        * [Lorem Ipsum](https://www.lipsum.com/)

### 一些免責聲明

* 由於本人只有碩班畢業，而且寫中文論文，所以這份Template就是只能產出**中文碩士**論文。
* 由於本人論文沒有附錄，所以**這份Template的「附錄」排版是沒有經過審查**的喔。
* 由於本人十分討厭在draft模式下無法看到圖片的狀況，所以本模板即使是在draft模式下也會將圖片載入（編譯時間較久）。
    > 如果圖片使用Microsoft Visio繪製，強烈建議在輸出時調整版面大小並選擇pdf檔案，且一樣用`\includegraphics`載入圖片，這樣可以大幅減少論文檔案大小，並加速編譯時間。
* 由於本人十分討厭看到在編譯時有waring這件事，所以本模板在**draft模式**有設置以下參數，這樣才不會一直出現`underfull hbox (badness 10000)`的warning。
    ```latex
    \hbadness 10000
    ```
* ~~由於陽明交大至今（Sep. 8, 2021）尚未決定校徽，故目前繳交之論文仍不需要附上浮水印，所以本模板沒有浮水印的功能。~~
* 感謝[@linroex](https://github.com/linroex)的持續[貢獻更新]([JingWangTW/NYCU-Thesis-Template#3](https://github.com/JingWangTW/NYCU-Thesis-Template/pull/3))，終於有校徽浮水印啦！目前不管哪一種設定都會自動帶出校徽。
* 學校僅針對封面與書名頁等非內文之內容有相關論文格式規範，有關內文之規範則由各研究系所訂定細則。由於本人為資科工碩畢業，所以基本上此Template應該是可以無痛通過資科工碩的系所審查，但是其餘科系真的不好說，QQ。

### 我不知道怎麼分類

* 有寫個GitHub Action啦，只要將專案Push Tag到GitHub就會觸發（Tag名稱須為`v*.*.*`），請GitHub幫你編譯並將編完的PDF發到Release。
    > 在編譯過程中會自動下載google標楷體，但是他的缺字很多，所以建議從 Windows 作業系統把標楷體字體幹出來，然後放在該專案根目錄並命名為 **`kaiu.ttf`**，在編譯過程會自動抓這個字體放進去。
* 本人是2021年畢業的，有些格式規範是後面新加的，所以這邊提供的範本可能不一定會跟上最新的規範。最近（Sep. 16th, 2023）因為比較閒，又剛好有貢獻者發了PR，所以就來順便更新一下，發現最近剛好要求要加入一份新的文件「著作彙編之學位論文資訊及彙編學術著作之共同作者貢獻聲明書」，就把它放進來啦。但是小弟也不會沒事在那邊看最新規範，所以請各位使用者還是要一下最新規範。

## 資料夾結構

```bash
NYCU-Thesis-Template
├── 0-Spine/                                    # 0-書脊（書背）（就是論文印出來後面的那一條啦）
│   ├── 書背範本_學校提供.pdf                      #   學校提供的範本
│   ├── 資科工碩書脊範本.pdf                       #   資科工碩提供的範本
├── 1-Authorization/                            # 1-授權書們 (將論文上傳至圖書館通過審核後，可以拿到的文件們)
│   ├── 1-Authorization.pdf                     #   將底下兩份文件合併後的檔案（可以在main.tex中提供給模板使用）
│   ├── Authorization.pdf                       #   電子檔著作權授權書（範本）
│   └── PostponePublicationApplication.pdf      #   延後公開申請書（範本）
├── 2-Approval/                                 # 2-論文審定同意書書 (在口試當天要給口委簽名的一份文件)
│   ├── 1-Approvale.pdf                         #   最後需要放進論文印出來的合併檔
│   ├── Approval_en.pdf                         #   論文審定書（英文範本）
│   ├── Approval_zh.pdf                         #   論文審定書（中文範本）
│   ├── 國立陽明交通大學論文審定同意書_中文版_110.07版.docx  #   論文審定書（中文範本）
│   ├── 國立陽明交通大學論文審定同意書_中文版_110.07版.odt   #   論文審定書（中文範本）
│   ├── 國立陽明交通大學論文審定同意書_英文版_110.07版.docx  #   論文審定書（英文範本）
│   └── 國立陽明交通大學論文審定同意書_英文版_110.07版.odt   #   論文審定書（英文範本）
├── 3-Acknowledgement/                          # 3-誌謝
│   ├── 1-Acknowledgement.tex                   #   誌謝，模板有提供`acknowledgement` enviroment可以使用。
├── 4-Abstracts/                                # 4-摘要們
│   ├── 1-Abstract_zh.tex                       #   中文摘要，模板有提供`zhAbstract` enviroment可以使用。
│   └── 2-Abstract_en.tex                       #   英文摘要，模板有提供`enAbstract` enviroment可以使用。
├── 5-Chapters/                                 # 5-論文各節內容 (可自行新增)
│   ├── 1-Introduction.tex                      #   這一份文件有提供切分段落的方法
│   ├── 2-RelatedWorks.tex                      #   這一份文件有提供如何\cite別人的文獻的方法
│   ├── 3-Design.tex                            #   這一份文件有提供如何在論文中插入公式和pseudo code段落的方法
│   ├── 4-Evaluation.tex                        #   這一份文件有提供如何插入表格和圖片的方法
│   └── 5-Conclusion.tex                        #   這一份文件有提供如何寫註腳的方法
├── 6-Reference/                                # 6-參考文獻
│   └── thesis.bib                              #   參考文獻檔，有提供一些常見的文獻引入範例
├── 7-Appx/                                     # 7-附錄 (可自行新增)
│   ├── 1-Data.tex                              #   就是一個附錄，模板有提供`Appx` environment可以使用。
│   └── 1-Spec.tex                               
├── 8-CoAuthor/                                 # 8-著作彙編之學位論文資訊及彙編學術著作之共同作者貢獻聲明書
│   ├── 1-CoAuthor.pdf                          #   最後需要放進論文印出來的合併檔
│   ├── CoAuthor_en.pdf                         #   貢獻聲明書（英文範本）
│   ├── CoAuthor_zh.pdf                         #   貢獻聲明書（中文範本）
│   ├── 著作彙編之學位論文資訊及彙編學術著作之共同作者貢獻聲明書_中文版_20230721.docx  #   貢獻聲明書（中文範本）
│   ├── 著作彙編之學位論文資訊及彙編學術著作之共同作者貢獻聲明書_中文版_20230721.odt   #   貢獻聲明書（中文範本）
│   ├── 著作彙編之學位論文資訊及彙編學術著作之共同作者貢獻聲明書_英文版_20230721.docx  #   貢獻聲明書（英文範本）
│   └── 著作彙編之學位論文資訊及彙編學術著作之共同作者貢獻聲明書_英文版_20230721.odt   #   貢獻聲明書（英文範本）      
├── Class/                                      # * 論文模板 (如果沒事不要亂動)
│   ├── NYCU-Thesis.cls                         #   國立陽明交通大學碩士中文論文模板
│   └── xCJKnumb.sty                            #   中文套件
├── Config/                                     # * 相關參數設定
│   ├── config.tex                              #   設定論文標題、作者資訊等
│   └── fonts.tex                               #   字型設定 (請拜託填一下你的系統字體名稱)
├── Figures/                                    # * 論文圖片 (可自行新增)
│   ├── watermark.png                           #   校徽浮水印本人
│   └── ...
├── Otherss/                                    # * 貼心的提供一些文件
│   └── 國立陽明交通大學博碩士學位論文格式規範_中英對照.pdf
├── .gitignore                                  # git檔案忽略清單
├── LICENSE                                     # 本專案授權
├── Makefile                                    # Linux makefile
├── README.md                                   # 說明文件 (本檔案)
├── WinMakefile                                 # Winodws makefile
├── latexmkrc                                   # 給overleaf看的編譯環境設定
└── main.tex                                    # 論文主要檔案
```

## 工商服務時間

* ~~[Overleaf](https://www.overleaf.com/) (The easy to use, online, collaborative LaTeX editor)~~（有回報 #7 Overleaf免費版好像編不完這個templeate惹 QQ）
* [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) (on [Visual Studio Code](https://code.visualstudio.com/))

## LaTex Reference

* [The LaTex Project](https://www.latex-project.org/)
* [CTAN](https://www.ctan.org/)
* [LaTeX Wikibooks](https://en.wikibooks.org/wiki/LaTeX)

## Acknowledgement

* 此模板修改自原[交大模板](https://github.com/yungshenglu/NCTU-Thesis-Template)
    * 原作者：[yungshenglu](https://github.com/yungshenglu)
    * 原模板專案：[yungshenglu/NCTU-Thesis-Template](https://github.com/yungshenglu/NCTU-Thesis-Template)

## Contribution

* 如果因為使用了這份模板有被擋格式，或是有甚麼最新的文件需要夾進論文裡，歡迎發Issue，我有空看到就會更新，或是更歡迎發PR造福後面的學弟妹。
* 或是也可以發個Issue，告訴大家你用了這份Template成功畢業惹。

## License

[GNU GENERAL PUBLIC LICENSE Version 3](LICENSE)
