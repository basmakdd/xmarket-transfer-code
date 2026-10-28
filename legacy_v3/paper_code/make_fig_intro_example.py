# Compact 2x2 layout for full text width in a 1-column manuscript.
# Canvas 1000 px wide: at 16 cm print width, 18 px text ~ 8 pt.
PW,PH,M,G=485,340,8,14
W=M*2+PW*2+G; H=M*2+PH*2+G
C=dict(de="#3B6FB6",mx="#E07A2E",good="#2E9E6A",bad="#D64545",ink="#2B2F36",sub="#5F6672",panel="#F3F4F7",line="#D2D7DE")
o=[];a=o.append
a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Liberation Sans, Arial, Helvetica, sans-serif">')
a(open(__import__("os").path.join(__import__("os").path.dirname(__file__),"fig_intro_icons_defs.svg"), encoding='utf-8').read())
a(f'<rect width="{W}" height="{H}" fill="#fff"/>')
def t(x,y,s,size=19,fill=C["ink"],w="normal",anchor="middle",it=False):
    st=' font-style="italic"' if it else ''
    a(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{w}" text-anchor="{anchor}"{st}>{s}</text>')
def icon(n,x,y,s=40):
    a(f'<circle cx="{x}" cy="{y}" r="{s*0.6}" fill="#fff" stroke="{C["line"]}" stroke-width="1.6"/>')
    a(f'<use href="#{n}" x="{x-s/2}" y="{y-s/2}" width="{s}" height="{s}"/>')
def badge(x,y,ok,r=19):
    a(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{C["good"] if ok else C["bad"]}"/>')
    k=r/22
    if ok: a(f'<path d="M{x-10*k} {y} L{x-3*k} {y+8*k} L{x+11*k} {y-8*k}" fill="none" stroke="#fff" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>')
    else: a(f'<path d="M{x-8*k} {y-8*k} L{x+8*k} {y+8*k} M{x+8*k} {y-8*k} L{x-8*k} {y+8*k}" stroke="#fff" stroke-width="4" stroke-linecap="round"/>')
def flag(x,y,c,w=27,h=18):
    if c=="de":
        for i,col in enumerate(["#1F1F1F","#D52B1E","#F2C200"]): a(f'<rect x="{x}" y="{y+i*h/3}" width="{w}" height="{h/3}" fill="{col}"/>')
    else:
        for i,col in enumerate(["#1F8A4C","#FFFFFF","#C8102E"]): a(f'<rect x="{x+i*w/3}" y="{y}" width="{w/3}" height="{h}" fill="{col}"/>')
        a(f'<circle cx="{x+w/2}" cy="{y+h/2}" r="2.5" fill="#8A6D3B"/>')
    a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="#C9CED6" stroke-width="1"/>')
def store(x,y,c,w=112):
    col=C[c]
    a(f'<rect x="{x}" y="{y+24}" width="{w}" height="64" rx="4" fill="#fff" stroke="{col}" stroke-width="2.5"/>')
    n=6; sw=w/n
    for i in range(n):
        a(f'<path d="M{x+i*sw} {y} H{x+(i+1)*sw} V{y+22} A{sw/2} {sw/4} 0 0 1 {x+i*sw} {y+22} Z" fill="{col if i%2==0 else "#fff"}" stroke="{col}" stroke-width="1.6"/>')
    a(f'<rect x="{x+w*0.12}" y="{y+42}" width="{w*0.38}" height="46" rx="3" fill="{col}" opacity="0.18"/>')
    a(f'<rect x="{x+w*0.58}" y="{y+42}" width="{w*0.30}" height="24" rx="3" fill="{col}" opacity="0.18"/>')
def panel(i,title):
    x=M+(i%2)*(PW+G); y=M+(i//2)*(PH+G)
    a(f'<g transform="translate({x},{y})">')
    a(f'<rect width="{PW}" height="{PH}" rx="14" fill="{C["panel"]}"/>')
    t(18,34,title,22,w="bold",anchor="start")
def end(): a('</g>')

# (a)
panel(0,"(a) Two stores, one catalogue")
store(40,54,"de"); store(PW-40-112,54,"mx")
a(f'<path d="M166 108 H{PW-172}" stroke="#6B7280" stroke-width="2.5" stroke-dasharray="6 5" marker-end="url(#arr)"/>')
t(PW/2,96,"borrow?",18,fill=C["sub"],it=True)
flag(40,158,"de"); t(74,174,"Germany",19,w="bold",anchor="start"); t(96,198,"many purchases",18,fill=C["sub"])
flag(PW-152,158,"mx"); t(PW-118,174,"Mexico",19,w="bold",anchor="start"); t(PW-96,198,"few purchases",18,fill=C["sub"])
a(f'<rect x="14" y="214" width="{PW-28}" height="56" rx="10" fill="#fff" stroke="{C["line"]}" stroke-width="1.6"/>')
t(28,248,"Same products",19,w="bold",anchor="start")
for i,n in enumerate(["camera","card","tripod","case","cable"]): icon(n,203+i*58,242,34)
a(f'<rect x="14" y="278" width="{PW-28}" height="50" rx="10" fill="#fff" stroke="{C["line"]}" stroke-width="1.6"/>')
t(28,309,"No shared shoppers",19,w="bold",anchor="start")
for i in range(3):
    a(f'<use href="#shopper" x="{214+i*26}" y="290" width="25" height="25" fill="{C["de"]}"/>')
    a(f'<use href="#shopper" x="{364+i*26}" y="290" width="25" height="25" fill="{C["mx"]}"/>')
a(f'<circle cx="335" cy="303" r="14" fill="none" stroke="{C["bad"]}" stroke-width="3.2"/><line x1="325" y1="313" x2="345" y2="293" stroke="{C["bad"]}" stroke-width="3.2"/>')
end()

# (b)
panel(1,"(b) Popularity stays local")
def bestlist(x,c,items,label):
    flag(x,58,c); t(x+34,74,label,19,w="bold",anchor="start")
    for i,(n,v) in enumerate(items):
        yy=118+i*62
        t(x+12,yy+7,f"#{i+1}",19,w="bold",fill=C["sub"])
        icon(n,x+62,yy,38)
        a(f'<rect x="{x+94}" y="{yy-10}" width="{v}" height="20" rx="4" fill="{C[c]}"/>')
bestlist(24,"de",[("camera",96),("tripod",74),("card",54)],"Germany")
bestlist(276,"mx",[("case",96),("cable",74),("card",54)],"Mexico")
a(f'<line x1="{PW/2+6}" y1="60" x2="{PW/2+6}" y2="262" stroke="{C["line"]}" stroke-width="2"/>')
badge(PW/2+6,180,False)
t(PW/2,300,"Best-sellers differ between countries:",19,w="bold")
t(PW/2,324,"German top items mislead in Mexico",18,fill=C["sub"])
end()

# (c)
panel(2,"(c) Shopping patterns travel")
def graph(x,c,label):
    flag(x+50,58,c); t(x+84,74,label,19,w="bold",anchor="start")
    cx=x+112
    pos={"camera":(cx,112),"card":(cx-62,172),"tripod":(cx+62,172),"case":(cx-44,238),"cable":(cx+44,238)}
    for u,v in [("camera","card"),("camera","tripod"),("case","cable")]:
        (x1,y1),(x2,y2)=pos[u],pos[v]
        a(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{C[c]}" stroke-width="5.5" stroke-linecap="round" opacity="0.85"/>')
    for n,(xx,yy) in pos.items(): icon(n,xx,yy,36)
graph(14,"de","Germany"); graph(PW-14-224,"mx","Mexico")
a(f'<line x1="{PW/2}" y1="60" x2="{PW/2}" y2="262" stroke="{C["line"]}" stroke-width="2"/>')
badge(PW/2,205,True)
t(PW/2,300,"Bought-together links mostly match:",19,w="bold")
t(PW/2,324,"camera → card, tripod;  phone case → cable",18,fill=C["sub"])
end()

# (d)
panel(3,"(d) What should Mexico borrow?")
a(f'<use href="#shopper" x="22" y="52" width="40" height="40" fill="{C["mx"]}"/>')
t(72,70,"A Mexican shopper just bought",19,anchor="start")
t(72,93,"a phone case. Using German data:",19,anchor="start")
icon("case",PW-52,74,38)
def row(y,title,note,items,ok):
    col=C["good"] if ok else C["bad"]
    a(f'<rect x="14" y="{y}" width="{PW-28}" height="100" rx="12" fill="#fff" stroke="{col}" stroke-width="2.2"/>')
    t(32,y+38,title,19,w="bold",anchor="start")
    t(32,y+68,note[0],18,anchor="start",fill=C["sub"])
    t(32,y+89,note[1],18,anchor="start",fill=C["sub"])
    for i,n in enumerate(items): icon(n,334+i*52,y+62,36)
    badge(PW-44,y+50,ok)
row(116,"Borrow best-sellers",("Suggests camera, tripod:","popular abroad, unrelated here"),["camera","tripod"],False)
row(226,"Borrow co-purchase patterns",("Suggests charging cable:","a link that also holds in Mexico"),["cable"],True)
end()
a('</svg>')
open("fig_intro_example.svg","w", encoding='utf-8').write("\n".join(o))
print(W,H)
