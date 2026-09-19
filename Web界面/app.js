/* ══════════════════════════════════════════════════════════════════════
   小方 Web · 界面逻辑
   网页这边只干三件事：
     1) 把用户打的字，原样送进本地那个引擎；
     2) 把引擎吐出来的东西，一个字一个字地画出来；
     3) 顺手管着心跳 —— 窗口一关，引擎自己就收摊。
   界面是"现代拟物"那一路的：只靠白色和阴影分凸凹，浮层用毛玻璃，
   不追 AI 网站那套。
   ══════════════════════════════════════════════════════════════════════ */
(() => {
'use strict';

const $  = s => document.querySelector(s);
const $$ = s => Array.from(document.querySelectorAll(s));
const sleep = ms => new Promise(r => setTimeout(r, ms));
const rnd = (a, b) => a + Math.random() * (b - a);
const REDUCED = !!(window.matchMedia &&
  window.matchMedia('(prefers-reduced-motion: reduce)').matches);

/* ── 全局状态 ───────────────────────────────────────────────────────── */
const S = {
  state: 'starting', detail: '',
  model: '', modelId: '', tier: '', legacy: false,
  catalog: [], flat: {}, degraded: false,
  think: { mode: 'think', turns: 3 },
  since: 0, beatMs: 1500, hold: 10,
  cid: '', convName: '',
  nearBottom: true, idleTimer: 0,
  progTimer: 0, prog: 0,
  lastUser: '', runlog: [], runCount: 0,
  lastEvt: 0, esWatch: 0, es: null,
  booted: false, plate: null, trainSteps: null
};
S.tierCn = k => ({ lite: 'Lite（最快）', pro: 'Pro（更聪明）',
                   ultra: 'Ultra（最深度）' }[k] || k || '—');

/* ── 小工具 ─────────────────────────────────────────────────────────── */
const esc = s => String(s).replace(/[&<>"]/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

/* ── 轻量 Markdown ────────────────────────────────────────────────────
   先整段转义(堵注入)，再按代码块切，非代码段走 mdBlock 处理块级
   (标题/列表/表格/引用/分割线)、mdInline 处理行内(粗体/行内码/链接)。 */
function fmt(text) {
  let out = '';
  esc(String(text == null ? '' : text)).split(/```/).forEach((p, i) => {
    if (i % 2 === 1) {
      const body = p.replace(/^[a-zA-Z0-9+#._-]*\r?\n/, '').replace(/\s+$/, '');
      out += '<pre><code>' + body + '</code></pre>';
    } else {
      out += mdBlock(p);
    }
  });
  return out;
}

function mdInline(s) {
  return s
    .replace(/`([^`\n]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>')
    .replace(/\[([^\]\n]+)\]\((\/(?:[^)\s]|&#x2F;)+)\)/g, '<b>$1</b>')
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g,
      '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
}

function mdBlock(s) {
  const lines = s.split('\n');
  const out = [];
  let para = [];
  const flush = () => {
    if (para.length) { out.push('<p>' + para.map(mdInline).join('<br>') + '</p>'); para = []; }
  };
  let i = 0, m;
  const n = lines.length;
  while (i < n) {
    const ln = lines[i];
    const t = ln.trim();
    if (t === '') { flush(); i++; continue; }
    const h = /^(#{1,6})\s+(.*)$/.exec(t);
    if (h) {
      flush(); out.push('<h' + h[1].length + '>' + mdInline(h[2]) + '</h' + h[1].length + '>');
      i++; continue;
    }
    if (/^(-{3,}|\*{3,}|_{3,})\s*$/.test(t)) { flush(); out.push('<hr>'); i++; continue; }
    const bq = /^>\s?(.*)$/.exec(t);
    if (bq) {
      flush(); const blk = '<blockquote>';
      let j = i;
      while (j < n && /^>\s?/.test(lines[j])) {
        out.push((j === i ? blk : '') + mdInline(lines[j].replace(/^>\s?/, '')));
        j++;
      }
      out.push('</blockquote>'); i = j; continue;
    }
    /* 表格：连续行里出现"|"，且第二行是分隔线才当表渲 */
    if (t.includes('|') && /^\s*\|?[\s:|-]+(\|[:\s-]*)?\s*$/.test(lines[i + 1] || '')) {
      flush();
      const cell = ln2 => mdInline(ln2.replace(/^\s*\|/, '').replace(/\|\s*$/, '')).split('|');
      const hdr = cell(lines[i]);
      i += 2;
      let body = '';
      while (i < n && lines[i].trim().includes('|')) { body += '<td>' + cell(lines[i]).join('</td><td>') + '</td>'; i++; }
      out.push('<table><thead><tr>' + hdr.map(x => '<th>' + x.trim() + '</th>').join('') +
               '</tr></thead><tbody>' + (body ? '<tr>' + body + '</tr>' : '') + '</tbody></table>');
      continue;
    }
    const ul = /^([-*+])\s+(.*)$/.exec(t);
    if (ul) {
      flush(); out.push('<ul>');
      while (i < n && (m = /^[-*+]\s+(.*)$/.exec(lines[i].trim()))) {
        out.push('<li>' + mdInline(m[1]) + '</li>'); i++;
      }
      out.push('</ul>');
      continue;
    }
    const ol = /^\d+[.、]\s+(.*)$/.exec(t);
    if (ol) {
      flush(); out.push('<ol>');
      while (i < n && (m = /^\d+[.、]\s+(.*)$/.exec(lines[i].trim()))) {
        out.push('<li>' + mdInline(m[1]) + '</li>'); i++;
      }
      out.push('</ol>');
      continue;
    }
    para.push(t);
    i++;
  }
  flush();
  return out.join('');
}

/* 返回体里带 net:true 就说明"根本不是这个桥接程序回的"，随便哪个版本的老 py
   都得先能认出来，别把"连不上"当成"接口报错" */
async function api(path, body, method) {
  try {
    /* 带 body 的默认 POST；不带 body 的这些"动作型"接口（开工新对话、退出）
       也得 POST——后端这些口子是 methods=["POST"]，用裸 GET 去敲它只会 404/405。 */
    const m = method || (body ? 'POST' : undefined);
    const r = await fetch(path, body
      ? { method: m, headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body) }
      : (m ? { method: m } : undefined));
    if (!r.ok) return { ok: false, http: r.status, err: 'HTTP ' + r.status };
    return await r.json();
  } catch (e) { return { ok: false, net: true, err: String(e) }; }
}

let toastTimer = 0, toastGone = 0;
function toast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  clearTimeout(toastGone); t.classList.remove('out');
  t.hidden = false;
  clearTimeout(toastTimer);
  /* 到点了先挂 .out 让它沉下去，等它走完再 hidden。
     直接 hidden 是"啪"一下凭空消失，跟前头那么讲究的入场完全接不上。 */
  toastTimer = setTimeout(() => {
    t.classList.add('out');
    toastGone = setTimeout(() => { t.hidden = true; t.classList.remove('out'); }, 280);
  }, 2400);
}

/* ── 状态灯 ─────────────────────────────────────────────────────────── */
const STATE_TXT = {
  starting: ['正在连引擎…', 'load'],
  loading:  ['正在把模型搬进内存…', 'load'],
  ready:    ['就绪', 'ok'],
  working:  ['思考中…', 'busy'],
  down:     ['引擎不在了', 'bad']
};

function setState(st, detail) {
  S.state = st; if (detail) S.detail = detail;
  const [txt, cls] = STATE_TXT[st] || [st, ''];

  const pill = $('#statePill'), conn = $('#conn');
  pill.className = 'state-pill ' + cls;
  conn.className = 'conn ' + cls;
  $('#stateTxt').textContent = txt;
  $('#connTxt').textContent = (st === 'down') ? '引擎不在了' : txt;

  $('#stopBtn').hidden = (st !== 'working');
  $('#sendBtn').disabled = (st === 'down');

  if (st === 'ready') { $('#brandSub').textContent = '本地跑 · 不上云'; }
  if (st === 'down')  { $('#brandSub').textContent = '引擎退出了'; }

  if (st === 'starting' || st === 'loading') showOverlay();
  else if (st === 'ready') finishOverlay();
  else if (st === 'down') { ovTitle('引擎退出了'); ovBar(100); setTimeout(hideOverlay, 1400); }
}

/* ── 遮罩（切模型 / 冷启动） ────────────────────────────────────────── */
function ovTitle(t, sub) {
  if (t) $('#ovTitle').textContent = t;
  if (sub !== undefined) $('#ovSub').textContent = sub;
}
function ovBar(p) {
  S.prog = p;
  $('#ovBar').style.width = p + '%';
  $('#ovArc').style.strokeDashoffset = String(339.3 * (1 - p / 100));
}
function showOverlay() {
  $('#overlay').hidden = false;
  if (S.prog === 0) {
    ovBar(2);
    clearInterval(S.progTimer);
    S.progTimer = setInterval(() => {
      /* 真实进度引擎不说，就让进度条往 92% 慢慢蹭，别装得太满 */
      const p = S.prog + Math.max(0.6, (92 - S.prog) * 0.055);
      ovBar(p >= 92 ? 92 : p);
    }, 320);
  }
}
function finishOverlay() {
  clearInterval(S.progTimer); S.progTimer = 0;
  ovBar(100);
  setTimeout(() => { hideOverlay(); ovBar(0); }, 420);
}
function hideOverlay() { $('#overlay').hidden = true; }

function ovLog(line) {
  const box = $('#ovLog');
  const d = document.createElement('div');
  d.textContent = line;
  box.appendChild(d);
  while (box.children.length > 3) box.removeChild(box.firstChild);
}

/* ── 消息流 ─────────────────────────────────────────────────────────── */
const stream = $('#stream');
const msgs = $('#msgs');

/* 首屏还露着脸吗？—— 后面有好几处都要问这一句，索性收成一个函数。 */
function helloOn() { const h = $('#hello'); return !!h && !h.hidden; }

function scrollBottom(force) {
  /* 首屏在场的时候先别滚。后台的系统日志是一行一行自己往里灌的，
     它要是照样把流滚到底，首屏连人带按钮就被顶到屏幕外头去了 ——
     看着像"按钮点不动"，其实是压根不在屏上。 */
  if (helloOn()) return;
  if (force || S.nearBottom) stream.scrollTop = stream.scrollHeight;
}
stream.addEventListener('scroll', () => {
  S.nearBottom = (stream.scrollHeight - stream.scrollTop - stream.clientHeight) < 90;
});

/* ══════════ 首屏那只正脸小方 ══════════ */
function hideHello() {
  const h = $('#hello');
  /* 先把滚动条放回来：这一步不能因为"已经藏过了"就跳过，
     不然滚动条会一直锁着，后面的对话就滚不动了 */
  stream.classList.remove('hello-on');
  if (h) h.hidden = true;
}

/* 每次重新露脸都要把入场动画重放一遍 —— 换新对话的时候，那一下"重新开场"
   的感觉全靠它 */
function showHello(animate) {
  const h = $('#hello'); if (!h) return;
  const wasHidden = h.hidden;
  if (animate === undefined) animate = wasHidden;
  h.hidden = false;
  /* 首屏在场：滚动条收掉。不收的话那 9px 会把首屏整体往左挤半格 */
  stream.classList.add('hello-on');
  shuffleChips();            /* v3.8: 四个快问每次都换一下出场顺序, 不会回回一样 */
  /* v3.8: 开机信号 —— 正常先让 0102 数字一个个冒出来, 再轮到标语打字;
     只有精简动效才直接敲标语(数字彩蛋一并省掉)。 */
  if (REDUCED) { leadType(); return; }
  runVSign();

  const card = $('#helloCard'), ghost = $('#helloGhost'), wrap = $('#ghostWrap');
  if (wrap) { wrap.style.transform = ''; }
  if (card) { card.style.transform = ''; card.style.animation = 'none';
              void card.offsetWidth; card.style.animation = ''; }
  if (ghost) { ghost.style.animation = 'none';
               void ghost.offsetWidth; ghost.style.animation = ''; }
  $$('.chip').forEach(c => { c.style.animation = 'none';
                            void c.offsetWidth; c.style.animation = ''; });
}

/* v3.8: 启动彩蛋 —— 打开那句"0 1 0 2 0 3 0 4"八位数字, 在卡片顶部一位一位跳进来
   (每 0.13s 亮一位, 像一段开机自检信号)。八位都亮了再轻轻整块淡出, 才轮到标语打字。
   走过一页的比例: 每回来都从头亮, 老用户看一遍认得, 新用户看一遍就知道这是开机小方。 */
function runVSign() {
  const sign = $('#vSign'); if (!sign) return;
  sign.classList.remove('dim');
  const spans = Array.from(sign.querySelectorAll('span'));
  spans.forEach(s => s.classList.remove('on'));
  if (REDUCED) { sign.classList.add('dim'); return; }   // 精简动效时, 标语由 showHello 直接敲
  let i = 0;
  const t0 = (globalThis.vSignSeq || 0) + 1; globalThis.vSignSeq = t0;
  const step = () => {
    if (t0 !== globalThis.vSignSeq) return;      // 被新一次的出场顶掉了
    if (i >= spans.length) {
      setTimeout(() => { sign.classList.add('dim'); }, 380);
      setTimeout(() => { leadType(); }, 620);
      return;
    }
    spans[i].classList.add('on');
    i++;
    setTimeout(step, 130);
  };
  setTimeout(step, 150);
}

/* v3.8: 四个快问随机起手 —— 按钮的编号是它自己的(0 1 … 0 4, 打给引擎照旧),
   只是出场顺序每次洗一回, 首屏不会像念稿一样回回一行行对上去。 */
function shuffleChips() {
  const chips = $('#chips'); if (!chips) return;
  const kids = Array.from(chips.children);
  for (let i = kids.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = kids[i]; kids[i] = kids[j]; kids[j] = tmp;
  }
  kids.forEach(k => chips.appendChild(k));
}

/* ── 首屏那句标语 ───────────────────────────────────────────────────── */
/* 攒一把短句，每次露脸随机挑一条。挑之前先记住上一条 —— 就一句标语，
   连着两次撞上同一句，一眼就能看出来。 */
const LEAD_LINES = [
  '我不在云上，就在你这台机器里转。',
  '断网也照跑，话一个字都不出这台机器。',
  '想到什么，直接往下说就行。',
  '慢一点没关系，反正我不催你。',
  '你说的话只落在本地硬盘上，谁也拿不走。',
  '写代码、理思路、翻译，都归我。',
  '不用挑时间，我一直在这儿等着。',
  '这台机器里没有别人，就咱俩。',
  '说不清楚也没事，我帮你捋一遍。',
  '今天想干点什么？',
  '电费算你的，算力算我的。',
  '我不联网，所以不会拿你的话去喂别人。',
  '卡住了就换个说法，我换个角度再想。',
  '长话短说还是慢慢聊，你定。',
  '文件丢给我，我来读。',
  '别客气，把最啰嗦的那个版本说给我听。',
  '要是有哪句不对，直接怼我。',
  '需要我闭嘴的时候，按停就行。',
  '先随便敲一句，热热身也好。',
  '深夜也一样，我不困。',
  '这台机器不吵，你也不用小声说话。'
];

let leadTimer = 0, leadAt = -1;

/* 标语比回话敲得快一点，但标点照样停半拍，每一下都带点抖 ——
   一条十来字的短句也得有自己的呼吸 */
function leadSpeed(chars, i) {
  const n = chars.length || 1;
  const ch = chars[i - 1] || '';
  const p = i / n;
  let d = 46 * (1.32 - 0.6 * Math.sin(Math.PI * Math.min(1, p * 1.02)));
  d = Math.max(20, d);
  if ('，,、：:'.indexOf(ch) >= 0) d += 150;
  if ('；;'.indexOf(ch) >= 0) d += 185;
  if ('。！？!?…'.indexOf(ch) >= 0) d += 330;
  if ('「」“”《》（）()【】'.indexOf(ch) >= 0) d += 60;
  if (/[a-zA-Z0-9]/.test(ch)) d *= 0.8;
  return Math.max(16, d * rnd(0.76, 1.34));
}

function leadType() {
  const el = $('#helloLead'), caret = $('#leadCaret'), hello = $('#hello');
  if (!el || !hello) return;
  clearTimeout(leadTimer);

  /* 先把上一轮的收尾状态摘干净：景深那层动画才会重新起 */
  hello.classList.remove('typing');
  if (caret) caret.classList.remove('off');

  let i = leadAt;
  while (i === leadAt && LEAD_LINES.length > 1) {
    i = Math.floor(Math.random() * LEAD_LINES.length);
  }
  leadAt = i;
  const line = LEAD_LINES[i];

  /* 不想看动效：整句直接摆上去 */
  if (REDUCED) {
    el.textContent = line;
    if (caret) caret.classList.add('off');
    hello.classList.add('typing');
    return;
  }

  const chars = Array.from(line);
  el.textContent = '';
  let k = 0;

  const step = () => {
    if (k >= chars.length) {
      if (caret) caret.classList.add('off');
      /* 敲完了才加这个类 —— 背景那只开始散景，卡片轻轻浮一层 */
      hello.classList.add('typing');
      return;
    }
    el.textContent += chars[k];
    k++;
    leadTimer = setTimeout(step, leadSpeed(chars, k));
  };
  leadTimer = setTimeout(step, 300);
}

function clearStream() {
  msgs.innerHTML = '';
  S.runlog = []; S.runCount = 0;
  showHello(true);
}

/* ══════════ 打字机（非线性） ══════════ */
/* v3.8: TW 增加 pause —— 输入框一聚焦, 底下正在打的字"停住不动、不清空", 失焦再续上。
   老行为是一想操作就整段没了, 用户看着像被打断；停着显然更绅士。 */
const TW = { skip: false, timer: 0, pause: false };
let twChain = Promise.resolve();
let twPauser = 0;        // 正在等待恢复的 typeOut 是否挂在暂停上（给 skip 用）+ 恢复计数
let twResumeSig = 0;     // 暂停/恢复 切换的世代号, 防止旧计时器把不该续的打完
let sendChain = Promise.resolve();   // v3.8: 发送队列链 —— 新消息在这里排队, 一条接一条发

function pauseType(p) {
  TW.pause = !!p;
  if (TW.pause) {
    twResumeSig++;
    clearTimeout(TW.timer);
  } else {
    twResumeSig++;
    /* 恢复时如果此刻没有待续的 typeOut 计时器, 就什么都不做；计时器在 tick 里自己会来 */
  }
}

/* 每个字的间隔不是一个常数：起手稳一下、中段往前赶、收尾再顿一顿，
   标点和换行额外停一拍，英文数字快些，最后再抖一点手感 */
function typeSpeed(chars, i) {
  const n = chars.length || 1;
  const ch = chars[i - 1] || '';
  const p = i / n;
  let d = 17 * (1.42 - 0.78 * Math.sin(Math.PI * Math.min(1, p * 1.04)));
  d = Math.max(6.5, d);
  if ('，,、：:'.indexOf(ch) >= 0) d += 76;
  if ('；;'.indexOf(ch) >= 0) d += 94;
  if ('。！？!?…'.indexOf(ch) >= 0) d += 165;
  if (ch === '\n') d += 115;
  if ('「」“”《》（）()【】'.indexOf(ch) >= 0) d += 32;
  if (/[a-zA-Z0-9]/.test(ch)) d *= 0.74;
  return Math.max(6, d * rnd(0.74, 1.34));
}

function typeOut(bubble, text) {
  const full = String(text == null ? '' : text);
  const chars = Array.from(full);
  const row = bubble.closest('.row');
  const ava = row ? row.querySelector('.ava') : null;

  bubble.classList.add('typing');
  const tx = document.createElement('span');
  tx.className = 'tw-tx';
  const caret = document.createElement('i');
  caret.className = 'tw-caret';
  bubble.appendChild(tx); bubble.appendChild(caret);
  if (ava) ava.classList.add('speaking');

  return new Promise(resolve => {
    let i = 0;
    const total = chars.length;
    const mySig = twResumeSig;

    const done = () => {
      clearTimeout(TW.timer);
      TW.pause = false;
      bubble.classList.remove('typing');
      bubble.innerHTML = fmt(full);
      if (ava) ava.classList.remove('speaking');
      addActs(bubble, full);
      scrollBottom(false);
      resolve();
    };

    if (REDUCED || total === 0) { done(); return; }

    const tick = () => {
      if (TW.skip || i >= total) { done(); return; }
      /* v3.8: 输入框聚焦 → 暂停。字已经打出来的保持原样、光标留在原地,
         不清空也不继续；失焦(twResumeSig 变更)后重新起个计时器接上。 */
      if (TW.pause) {
        if (mySig === twResumeSig) {
          TW.timer = setTimeout(tick, 90);   // 还停在暂停期, 轻轻地再等等
        }
        return;
      }
      const left = total - i;
      let take = 1;
      /* 偶尔一连吐出两三个字 —— 匀着敲反而像机器在念稿 */
      if (left > 14 && Math.random() < 0.17) take = 2 + (Math.random() * 2 | 0);
      i = Math.min(total, i + take);
      tx.textContent = chars.slice(0, i).join('');
      scrollBottom(false);
      /* v3.8: 用世代号洗澡 —— 暂停/恢复切换(twResumeSig++)后, 旧计时器即便还挂着
         也会因为 mySig !== twResumeSig 在下一跳直接停, 由新世代接着打。 */
      if (mySig !== twResumeSig) return;
      TW.timer = setTimeout(tick, typeSpeed(chars, i));
    };
    tick();
  });
}

/* 引擎可能一句分几段吐。新的来了就先把上一段收干净，别两条一起长 */
function queueType(bubble, text) {
  TW.skip = true;
  twChain = twChain
    .then(() => { TW.skip = false; return typeOut(bubble, text); })
    .catch(() => {});
  return twChain;
}

/* 点一下正在吐的那条，或者按 Esc，就一口气给完 */
msgs.addEventListener('click', e => {
  if (e.target.closest('.bubble.typing')) TW.skip = true;
});

/* 回答底下那两个小动作 */
function addActs(bubble, text) {
  const old = bubble.querySelector('.acts'); if (old) old.remove();
  const acts = document.createElement('div');
  acts.className = 'acts';

  const cp = document.createElement('button');
  cp.className = 'act'; cp.type = 'button'; cp.textContent = '复制';
  cp.addEventListener('click', async e => {
    e.stopPropagation();
    try { await navigator.clipboard.writeText(text); toast('抄好了'); }
    catch (err) { toast('这个窗口不让写剪贴板，自己选一下'); }
  });

  const sv = document.createElement('button');
  sv.className = 'act'; sv.type = 'button'; sv.textContent = '存成文件';
  sv.addEventListener('click', e => { e.stopPropagation(); saveText(text); });

  acts.appendChild(cp); acts.appendChild(sv);
  bubble.appendChild(acts);
  wireCodes(bubble);
}

/* 代码块上面的两个小按钮：复制 / 存进产物。
   addActs 在打字收尾和瞬时渲染两处都会调，所以所有代码块都覆盖到。 */
function wireCodes(bubble) {
  if (!bubble || !bubble.querySelector) return;
  bubble.querySelectorAll('pre').forEach(pre => {
    if (pre.__wired) return;
    pre.__wired = true;
    const code = pre.querySelector('code');
    const txt = (code ? code.textContent : pre.textContent) || '';
    const bar = document.createElement('div');
    bar.className = 'code-acts';
    const mk = (label, fn) => {
      const b = document.createElement('button');
      b.type = 'button'; b.textContent = label;
      b.addEventListener('click', e => { e.stopPropagation(); fn(); });
      bar.appendChild(b);
      return b;
    };
    const isLong = txt.split('\n').length > 40;
    mk('复制', async () => {
      try { await navigator.clipboard.writeText(txt); toast('抄好了'); }
      catch (_) { toast('这个窗口不让写剪贴板，自己选一下'); }
    });
    mk(isLong ? '存成产物 · ' + (extOf(txt) || '代码') : '存成产物', () => saveArtifact(txt));
    if (pre.nextSibling) pre.parentNode.insertBefore(bar, pre.nextSibling);
    else pre.parentNode.appendChild(bar);
  });
}

/* 从代码头几行猜一个扩展名，存进产物时用 */
function extOf(t) {
  if (/^import\s|^from\s|^def\s|^class\s|^#!|^if __name__|^async def/.test(t)) return 'py';
  if (/^#include|using namespace|std::|\blib\s*<|^template/.test(t)) return 'cpp';
  if (/^<!DOCTYPE|<html/.test(t)) return 'html';
  if (/^@import|^[a-zA-Z][\w-]*\s*\{|^\.[a-zA-Z_][\w-]*\s*\{|^#[\w-]+/.test(t)) return 'css';
  if (/^\/\/\s*|^<?xml|^package\s+|^public\s+class/.test(t)) return 'java';
  if (/^\s*(function\s+\w+\s*\(|const\s+\w+\s*=\s*\(|document\.|console\.|window\.|=>\s*\{?)/.test(t)) return 'js';
  if (/^--.*$|^CREATE\s+TABLE|^SELECT\s|^INSERT\s|^UPDATE\s/i.test(t)) return 'sql';
  return 'txt';
}

async function saveArtifact(text) {
  const r = await api('/api/artifact/save', { text: text, ext: extOf(text) }, 'POST');
  if (r && r.ok) { toast('已存进产物 · ' + r.name); artLoad(true); }
  else toast((r && r.err) || '没存进去');
}

function saveText(text) {
  const d = new Date(), p = n => String(n).padStart(2, '0');
  const name = '小方_' + d.getFullYear() + p(d.getMonth() + 1) + p(d.getDate()) +
               '_' + p(d.getHours()) + p(d.getMinutes()) + p(d.getSeconds()) + '.txt';
  try {
    const url = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }));
    const a = document.createElement('a');
    a.href = url; a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
    toast('存到"下载"里了');
  } catch (e) { toast('存不下来'); }
}

/* ══════════ 气泡 ══════════ */
/* 这条气泡在第几条？只数带 .bubble 的 .row（思考盒、系统行都不算），
   正好就是后端 CONV["msgs"] 里的下标 —— 两侧不必打架。 */
function bubbleIdx(row) {
  let i = 0;
  for (const r of msgs.querySelectorAll('.row')) {
    if (r === row) return i;
    if (r.querySelector('.bubble')) i++;
  }
  return -1;
}

function bubbleRow(who, text, opts) {
  opts = opts || {};
  hideHello();
  const row = document.createElement('div');
  row.className = 'row ' + (who === 'me' ? 'me' : 'ai');

  if (who === 'me') {
    /* 用户那头不摆小方的脸，用一枚墨色小圆牌 */
    const ava = document.createElement('div');
    ava.className = 'ava me-ava';
    ava.textContent = '你';
    row.appendChild(ava);
  } else {
    const ava = document.createElement('img');
    ava.className = 'ava';
    ava.src = 'assets/xf.jpg';
    ava.alt = '';
    row.appendChild(ava);
  }

  const b = document.createElement('div');
  b.className = 'bubble';
  row.appendChild(b);

  if (who === 'me') {
    /* 用户气泡右上角那三颗小工具：重发 / 删除 / 撤回。
       原始文本塞在 row._txt 上（getter 属性，不进 innerHTML，不会被 XSS）。 */
    row._txt = String(text == null ? '' : text);
    const tools = document.createElement('div');
    tools.className = 'bub-tools';
    const mk = (glyph, title, op) => {
      const bt = document.createElement('button');
      bt.type = 'button'; bt.title = title; bt.textContent = glyph;
      bt.addEventListener('click', e => {
        e.stopPropagation();
        if (op === 'resend') { send(row._txt); return; }
        const idx = bubbleIdx(row);
        if (idx < 0) { toast('这条找不到了'); return; }
        api('/api/history/edit', { op, idx }, 'POST').then(r => {
          if (r && r.ok === false) toast(r.err || '没改掉');
        });
      });
      return bt;
    };
    tools.appendChild(mk('↻', '重新发送这条', 'resend'));
    tools.appendChild(mk('✕', '删除此条消息', 'del'));
    tools.appendChild(mk('⏪', '撤回到本次发出', 'revoke'));
    row.appendChild(tools);
  }

  msgs.appendChild(row);
  scrollBottom(who === 'me');

  if (who === 'me' || opts.instant || opts.plain) {
    /* 翻旧对话时不放动画，一屏直接落下来 */
    b.innerHTML = fmt(String(text == null ? '' : text));
    if (!opts.plain) addActs(b, String(text == null ? '' : text));
  } else {
    queueType(b, text);
  }
  return b;
}

function thinkRow(text) {
  hideHello();
  const row = document.createElement('div');
  row.className = 'row ai';
  const pad = document.createElement('div');
  pad.style.width = '32px'; pad.style.flex = 'none';
  const d = document.createElement('details');
  d.className = 'thinkbox';
  const n = (S.runCount = S.runCount + 1);
  d.innerHTML = '<summary>· 思考过程 ' + n + '</summary><div class="tb-body"></div>';
  d.querySelector('.tb-body').innerHTML = fmt(text);
  row.appendChild(pad); row.appendChild(d);
  msgs.appendChild(row);
  scrollBottom(false);
}

function sysRow(text, cls) {
  const d = document.createElement('div');
  d.className = 'sysline ' + (cls || '');
  d.textContent = text;
  msgs.appendChild(d);
  scrollBottom(false);
}

/* 非 [ 开头的裸日志：收进流末尾一个折叠块里，不污染对话 */
function runLogRow(text) {
  S.runlog.push(text);
  let box = msgs.querySelector('.runlog-box');
  if (!box) {
    box = document.createElement('details');
    box.className = 'thinkbox runlog-box';
    box.innerHTML = '<summary></summary><div class="tb-body"></div>';
    msgs.appendChild(box);
  }
  box.querySelector('summary').textContent = '· 引擎原始输出 ' + S.runlog.length + ' 行';
  const body = box.querySelector('.tb-body');
  body.textContent = S.runlog.slice(-400).join('\n');
  if (box.open) scrollBottom(false);
}

/* ── 思考强度滑块 ───────────────────────────────────────────────────── */
const MODE_CN = { off: '不思考', think: '思考', multi: '多轮' };
const THINK_MODES = ['off', 'think', 'multi'];   // 三个停靠点，没有中间态

const thinkEl = $('#think');
const trackEl = $('#thinkTrack');
const marksEl = $('#thinkMarks');

/* 圆球的直径、轨道的内边距：跟 style.css 里 .think-knob / .think-track 的值一对一对齐，
   改尺寸只改这两行，JS 算出的位移和 CSS 画出来的不会打架。 */
const KNOB = 22;
const TRACK_PAD = 2;

function thinkSegs() { return $$('#thinkTrack .think-seg'); }

/* 角标那一排（不思考 / 思考 / 多轮）—— 轨道上沿的提示 + 直点入口 */
function thinkMarks() { return marksEl ? $$('#thinkMarks span') : []; }

/* 把「蓝色轨迹画到哪儿」换算成 0~1 的比例丢给 CSS。
   轨迹在 CSS 里是整条铺满、再用 scaleX 压出来的（这样不碰 width，
   不触发排版），所以这里给的是比例，不是像素宽度。
   cx 是旋钮中心的横坐标；轨道左右各留 TRACK_PAD 的 inset，得减掉。 */
function thinkFill(cx) {
  if (!trackEl) return;
  const inner = trackEl.clientWidth - TRACK_PAD * 2;
  const r = inner > 0 ? (cx - TRACK_PAD) / inner : 0;
  trackEl.style.setProperty('--fillR', Math.max(0, Math.min(1, r)).toFixed(4));
}

/* 旋钮停到第 i 档：球心对着那一格的中心 */
function thinkPlace(i) {
  const segs = thinkSegs();
  const seg = segs[i];
  const knob = $('#thinkKnob');
  if (!seg || !knob) return;
  const kw = KNOB;
  const x = seg.offsetLeft + seg.offsetWidth / 2 - kw / 2;
  knob.style.width = kw + 'px';
  knob.style.height = kw + 'px';
  knob.style.transform = 'translateX(' + x + 'px)';
  /* 蓝色的那截轨迹画到球心为止 */
  thinkFill(x + kw / 2);
}

function paintThink() {
  let idx = THINK_MODES.indexOf(S.think.mode);
  if (idx < 0) { idx = 1; S.think.mode = 'think'; }
  thinkSegs().forEach((s, k) => s.classList.toggle('on', k === idx));
  thinkMarks().forEach((m, k) => m.classList.toggle('on', k === idx));
  thinkPlace(idx);
  /* 多轮：整条轨道染蓝、缓缓发亮；轮数那枚小控件这时候才长出来 */
  const multi = S.think.mode === 'multi';
  thinkEl.classList.toggle('multi', multi);
  $('#turns').classList.toggle('on', multi);
  $('#turnsNum').textContent = S.think.turns;
  trackEl.setAttribute('aria-valuenow', String(idx));
  trackEl.setAttribute('aria-valuetext', MODE_CN[S.think.mode] || '');
}

/* 落点 → 最近的那一档（磁吸的唯一依据） */
function thinkNearest(clientX) {
  const segs = thinkSegs();
  const r = trackEl.getBoundingClientRect();
  const x = clientX - r.left;
  let best = 0, bd = Infinity;
  segs.forEach((s, i) => {
    const d = Math.abs(s.offsetLeft + s.offsetWidth / 2 - x);
    if (d < bd) { bd = d; best = i; }
  });
  return best;
}

/* 拖动途中：圆球贴着手指走，文字档位实时跟着最亮 */
function thinkFollow(clientX) {
  const segs = thinkSegs();
  const knob = $('#thinkKnob');
  const i = thinkNearest(clientX);
  const kw = KNOB;
  const x = clientX - trackEl.getBoundingClientRect().left;
  const maxX = trackEl.clientWidth - TRACK_PAD - kw;
  const px = Math.min(maxX, Math.max(TRACK_PAD, x - kw / 2));
  knob.style.width = kw + 'px';
  knob.style.height = kw + 'px';
  knob.style.transform = 'translateX(' + px + 'px)';
  thinkFill(px + kw / 2);
  segs.forEach((s, k) => s.classList.toggle('on', k === i));
  thinkMarks().forEach((m, k) => m.classList.toggle('on', k === i));
}

function thinkSet(mode) {
  if (THINK_MODES.indexOf(mode) < 0) mode = 'think';
  const changed = S.think.mode !== mode;
  S.think.mode = mode;
  paintThink();
  api('/api/think', { mode: mode, turns: S.think.turns }).then(r => {
    if (changed && r && r.ok) {
      toast('思考强度 · ' + MODE_CN[mode] + (mode === 'multi' ? ' ' + S.think.turns + ' 轮' : ''));
    }
  });
}

let thinkDrag = null;

trackEl.addEventListener('pointerdown', ev => {
  /* 整条轨道随便按哪儿都能拖：三格文字已经退化成不吃事件的量尺，
     所以这里不用再区分「点档位」还是「拖滑块」——松手时统一磁性归位。 */
  thinkDrag = ev.pointerId;
  thinkEl.classList.add('dragging');
  try { trackEl.setPointerCapture(ev.pointerId); } catch (_) {}
  thinkFollow(ev.clientX);
  ev.preventDefault();
});

trackEl.addEventListener('pointermove', ev => {
  if (thinkDrag === null || ev.pointerId !== thinkDrag) return;
  thinkFollow(ev.clientX);
});

function thinkDrop(ev) {
  if (thinkDrag === null || ev.pointerId !== thinkDrag) return;
  const seg = thinkSegs()[thinkNearest(ev.clientX)];
  thinkDrag = null;
  thinkEl.classList.remove('dragging');
  try { trackEl.releasePointerCapture(ev.pointerId); } catch (_) {}
  thinkSet(seg ? seg.dataset.mode : S.think.mode);   // 松手回弹归位
}
trackEl.addEventListener('pointerup', thinkDrop);
trackEl.addEventListener('pointercancel', thinkDrop);

/* 轨道上沿那排角标：点哪句就跳到哪一档 */
thinkMarks().forEach(m => m.addEventListener('click', () => thinkSet(m.dataset.mode)));

/* 左右方向键在三个停靠点之间挪，Home / End 直接跳到两头 */
trackEl.addEventListener('keydown', ev => {
  const cur = Math.max(0, THINK_MODES.indexOf(S.think.mode));
  let next = -1;
  if (ev.key === 'ArrowRight' || ev.key === 'ArrowUp') next = Math.min(THINK_MODES.length - 1, cur + 1);
  else if (ev.key === 'ArrowLeft' || ev.key === 'ArrowDown') next = Math.max(0, cur - 1);
  else if (ev.key === 'Home') next = 0;
  else if (ev.key === 'End') next = THINK_MODES.length - 1;
  if (next < 0 || next === cur) return;
  ev.preventDefault();
  thinkSet(THINK_MODES[next]);
});

$('#turnsMinus').addEventListener('click', e => {
  e.stopPropagation();
  S.think.turns = Math.max(1, S.think.turns - 1); paintThink();
  api('/api/think', { mode: 'multi', turns: S.think.turns });
});
$('#turnsPlus').addEventListener('click', e => {
  e.stopPropagation();
  S.think.turns = Math.min(8, S.think.turns + 1); paintThink();
  api('/api/think', { mode: 'multi', turns: S.think.turns });
});

/* ── 模型上拉菜单（三级） ───────────────────────────────────────────── */
const pop = $('#pop');
const tip = $('#tip');

/* v3.8.1: Covi 系(现役正式版)行是否用专属 logo —— id 以 covi 开头即是 */
function isCovi(it) { return /^covi/i.test(String(it.id || '')); }

function dotCls(it) { return it.ready ? 'mi-dot' : 'mi-dot no'; }

function miNode(it, colIdx) {
  const isLeaf = !!it.engine;
  const b = document.createElement('button');
  b.className = 'mi' + (!isLeaf ? ' parent' : '');
  if (it.id === S.modelId) b.classList.add('sel');

  let html = '';
  /* Covi 系列：左上角换成专属 logo(新形象)；其它仍按原来的圆点/缩进 */
  if (isCovi(it)) html += '<img class="mi-logo" src="assets/xf.webp" alt="">';
  else if (isLeaf) html += '<i class="' + dotCls(it) + '"></i>';
  html += '<span class="mi-nm">' + esc(it.label || it.id) + '</span>';
  if (it.badge) html += '<span class="mi-tag' + (it.badge === '备案' ? ' gray' : '') + '">' +
                        esc(it.badge) + '</span>';
  else if (!isLeaf && it.children) html += '<span class="mi-sub">' +
                        it.children.length + ' 个</span>';
  if (!isLeaf) html += '<svg viewBox="0 0 24 24" class="ic mi-arrow"><path d="M9 6l6 6-6 6"/></svg>';
  b.innerHTML = html;

  b.addEventListener('mouseenter', () => {
    $$('.mi.hot').forEach(x => x.classList.remove('hot'));
    b.classList.add('hot');
    showTip(b, it);
    if (colIdx === 0) { fillItems(it); clearKids(); }
    else if (colIdx === 1) { fillKids(it); }
  });
  b.addEventListener('click', () => {
    if (isLeaf) { chooseModel(it); return; }
    if (colIdx === 0) { fillItems(it); clearKids(); }
    else if (colIdx === 1) { fillKids(it); }
  });
  return b;
}

function fillList(box, nodes, colIdx) {
  box.innerHTML = '';
  if (!nodes || !nodes.length) return;
  nodes.forEach(n => box.appendChild(miNode(n, colIdx)));
}
function fillItems(group) { fillList($('#popItems'), group ? (group.items || []) : [], 1); }
function clearKids() { $('#popKids').innerHTML = ''; }
function fillKids(item) {
  clearKids();
  if (!item || !item.children) return;
  fillList($('#popKids'), item.children, 2);
}

function buildPop() {
  if (!S.catalog.length) {
    hidePop();
    toast(S.degraded ? '这个版本的桥接程序没给出模型清单' : '清单是空的');
    return;
  }
  fillList($('#popGroups'), S.catalog, 0);
  clearKids(); $('#popItems').innerHTML = '';
  if (S.catalog.length) fillItems(S.catalog[0]);
}

function showPop() {
  pop.hidden = false;
  $('#modelBtn').classList.add('open');
  const r = $('#modelBtn').getBoundingClientRect();
  pop.style.visibility = 'hidden'; pop.style.left = '0px'; pop.style.top = 'auto';
  pop.style.bottom = 'auto';
  const pw = pop.offsetWidth, ph = pop.offsetHeight;
  let left = r.left;
  if (left + pw > window.innerWidth - 14) left = Math.max(14, window.innerWidth - pw - 14);
  /* v3.8.1: 「上拉菜单」就该贴着输入框顶上 —— 底边固定(给输入框留 10px 呼吸)，
     内容一多整体往上长(顶边向上扩), 而不是贴窗口顶撒出去(那会让人以为跟输入框没关系)。 */
  const gap = 10;
  const popBottomY = r.top - gap;              // 弹窗底边想落在的像素 Y
  const wantTopY = popBottomY - ph;            // 相应顶边的像素 Y
  if (wantTopY >= 10) {
    pop.style.bottom = (window.innerHeight - popBottomY) + 'px';
    pop.style.top = 'auto';
  } else {
    pop.style.top = '10px'; pop.style.bottom = 'auto';   // 极矮窗口兜底, 起码别被裁
  }
  pop.style.left = left + 'px';
  pop.style.visibility = '';
}
function hidePop() {
  pop.hidden = true; tip.hidden = true;
  $('#modelBtn').classList.remove('open');
}

$('#modelBtn').addEventListener('click', e => {
  e.stopPropagation();
  if (!pop.hidden) { hidePop(); return; }
  /* buildPop 只负责把列表灌满（清单空的时候它自己会 hidePop + 吐个 toast），
     摆不摆出来是这里说了算 —— 之前拿"面板还隐不隐藏"当开关，
     那个值从来就没被 buildPop 翻过面，所以这菜单一次都弹不出来。 */
  buildPop();
  if (S.catalog.length) showPop();
});
document.addEventListener('click', e => {
  if (!pop.hidden && !pop.contains(e.target) && !$('#modelBtn').contains(e.target)) hidePop();
});

/* 悬停状态小卡 */
function showTip(anchor, it) {
  const isLeaf = !!it.engine;
  let html = '<h4>' + esc(it.label || it.id) + '</h4>';
  if (it.desc) html += '<div class="tp-desc">' + esc(it.desc) + '</div>';
  html += '<dl>';
  if (it.params) html += '<dt>体量</dt><dd>' + esc(it.params) + '</dd>';
  if (it.tier)   html += '<dt>档位</dt><dd>' + esc(S.tierCn(it.tier)) + '</dd>';
  if (isLeaf) {
    html += '<dt>本地缓存</dt><dd>' + esc(it.size || '—') + '</dd>';
    html += '<dt>状态</dt><dd>' + (it.ready ? '能秒开' : '要现算') + '</dd>';
  } else if (it.children) {
    html += '<dt>下面还有</dt><dd>' + it.children.length + ' 个</dd>';
  }
  html += '</dl>';
  if (it.note) html += '<div class="tp-note' + (isLeaf && it.ready ? ' ok' : '') + '">' +
                       esc(it.note) + '</div>';
  tip.innerHTML = html;
  tip.hidden = false;

  const r = anchor.getBoundingClientRect();
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  let left = r.right + 10;
  if (left + tw > window.innerWidth - 10) left = r.left - tw - 10;
  if (left < 10) left = 10;
  let top = r.top - 6;
  if (top + th > window.innerHeight - 10) top = window.innerHeight - th - 10;
  if (top < 10) top = 10;
  tip.style.left = left + 'px';
  tip.style.top = top + 'px';
}

/* 选中一个模型／让"点选即重启"的心跳别转太快 */
let modelBusy = false;
async function chooseModel(it) {
  hidePop(); tip.hidden = true;
  if (it.id === S.modelId) { toast('已经在用这个了'); return; }
  if (modelBusy) return;
  modelBusy = true;
  S.modelId = it.id;
  ovTitle('正在切换到 ' + (it.label || it.id) + ' …',
          it.ready ? '这一档本地有缓存，马上就好。'
                   : '本地没缓存，第一次要现算一会儿，之后就秒开了。');
  ovBar(0); S.prog = 0;
  showOverlay();
  clearStream();
  const r = await api('/api/model', { id: it.id });
  if (r && r.ok === false) { toast(r.err || '切不过去'); hideOverlay(); }
  modelBusy = false;
}

/* ── 历史对话 ───────────────────────────────────────────────────────── */
function timeAgo(ts) {
  if (!ts) return '';
  const d = new Date(ts * 1000), dt = Date.now() / 1000 - ts;
  if (dt < 60) return '刚刚';
  if (dt < 3600) return Math.floor(dt / 60) + ' 分钟前';
  if (dt < 86400) return Math.floor(dt / 3600) + ' 小时前';
  if (dt < 86400 * 7) return Math.floor(dt / 86400) + ' 天前';
  return (d.getMonth() + 1) + '月' + d.getDate() + '日';
}

function setConvCount(n) {
  const el = $('#convCount');
  if (!el) return;
  const last = Number(el.dataset.n || 0);
  el.textContent = n;
  el.dataset.n = n;
  if (n > last) { el.classList.remove('pop'); void el.offsetWidth; el.classList.add('pop'); }
}

async function refreshConvs() {
  const r = await api('/api/history');
  const box = $('#convList');
  box.innerHTML = '';
  if (!r || r.ok === false || !r.items) {
    setConvCount(0);
    box.innerHTML = '<div class="conv-empty">' +
      (r && r.net ? '口子还没通，历史过一会儿再读。'
                  : '这个版本的桥接程序还没有历史接口。<br>对话照样能聊，只是不会存档。') +
      '</div>';
    return;
  }
  const items = r.items || [];
  setConvCount(items.length);
  if (!items.length) {
    box.innerHTML = '<div class="conv-empty">还没有历史对话。<br>聊过的会自动存在本地，' +
                    '只留你说的和小方答的，思考过程一概不留。</div>';
    return;
  }
  items.forEach(it => {
    const b = document.createElement('button');
    b.className = 'conv' + (it.cid === S.cid ? ' on' : '');
    b.innerHTML = '<span class="nm"></span><span class="tm"></span>' +
      '<span class="del" title="删掉这条"><svg viewBox="0 0 24 24" class="ic">' +
      '<path d="M6 6l12 12M18 6L6 18"/></svg></span>';
    b.querySelector('.nm').textContent = it.name || '未命名';
    b.querySelector('.tm').textContent = timeAgo(it.updated) +
      (it.n ? ' · ' + it.n + ' 句' : '');
    b.addEventListener('click', async e => {
      if (e.target.closest('.del')) return;
      const o = await api('/api/history/open', { cid: it.cid });
      if (o && o.ok === false) { toast(o.err || '打不开'); return; }
      S.cid = it.cid;
      refreshConvs();
      $('.conv.on') && $('.conv.on').scrollIntoView({ block: 'nearest' });
    });
    b.querySelector('.del').addEventListener('click', async e => {
      e.stopPropagation();
      await api('/api/history/del', { cid: it.cid });
      if (it.cid === S.cid) S.cid = '';
      refreshConvs();
      toast('删掉了');
    });
    box.appendChild(b);
  });
}

async function newChat() {
  const r = await api('/api/history/new', null, 'POST');
  if (r && r.ok === false) { toast(r.err || '开不了新对话'); return; }
  S.cid = (r && r.cid) || S.cid;
  clearStream();
  S.nearBottom = true;
  stream.scrollTop = 0;
  refreshConvs();
  showHello(true);
  input.value = ''; autoGrow(); input.focus();
  toast('新的一页');
}
$('#newChat').addEventListener('click', newChat);

/* ── 发送 ───────────────────────────────────────────────────────────── */
const input = $('#input');

function autoGrow() {
  input.style.height = 'auto';
  input.style.height = Math.min(190, input.scrollHeight) + 'px';
}
input.addEventListener('input', autoGrow);
input.addEventListener('focus', () => { $('.composer').classList.add('focus'); hintStop(); pauseType(true); });
input.addEventListener('blur',  () => { $('.composer').classList.remove('focus'); hintStart(); pauseType(false); });
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
});

async function send(text) {
  const t = (text !== undefined ? text : input.value).trim();
  if (!t) return;
  if (S.state === 'down') { toast('引擎已经退出了，重启一下小方吧'); return; }
  if (text === undefined) { input.value = ''; autoGrow(); }

  /* v3.8: 排队机制 —— 引擎是"一次只答一句"的, 你连发好几条它得按先后一条条来。
     这里把"发出"塞进一条串行链: 上一条还在等引擎回, 这一条就先排着, 一行接一行摆进
     输入区上沿的队列气泡里; 等到前面全答完, 这条自己接着发出去。可无限排队。
     不想排队就点停止 —— 会把队列清空。 */
  sendChain = sendChain.then(async () => {
    try {
      await doCompose(t);
    } finally { /* 队列前进 */ }
  });
  return sendChain;
}

/* 真正把一句送进引擎并画上屏（send 的实质，排队链里逐条调用） */
async function doCompose(t) {
  hideHello();
  bubbleRow('me', t);      /* 先画上，引擎回头会把这句原样抄回来，重复的丢掉 */
  S.lastUser = t;
  const r = await api('/api/say', { text: t });
  if (r && r.ok === false) toast(r.err || '没送出去');
  refreshConvs();
}
$('#sendBtn').addEventListener('click', () => send());
$('#stopBtn').addEventListener('click', () => { TW.skip = true; api('/api/stop', null, 'POST'); toast('打断了'); });
$$('.chip').forEach(c => c.addEventListener('click', () => send(c.dataset.t)));

$('#quitBtn').addEventListener('click', async () => {
  if (!confirm('退出小方？本地那个推理进程也会一起关掉。')) return;
  await api('/api/quit', null, 'POST');
  document.body.innerHTML =
    '<div style="display:flex;height:100vh;align-items:center;justify-content:center;' +
    'font:15px/1.9 var(--sans,sans-serif);color:#4d5361;text-align:center">' +
    '小方已经退下了。<br><span style="font-size:12.5px;color:#8a8375">' +
    '这个窗口可以关掉了。</span></div>';
});
$('#sideToggle').addEventListener('click', () => {
  const app = $('#app');
  const willCollapse = !app.classList.contains('collapsed');
  /* 这一步是要把左边重新拉出来 —— 那右边那份就得让位（两边只留一个）。
     反过来，收左边这一下不用管右边，它本来就是收着的。 */
  if (!willCollapse) artClose();
  /* 用 willCollapse 定死，不用 toggle()：artClose() 刚才为了"还回左边原来的样子"
     已经动过 collapsed 这个类了，再 toggle() 就会把它的结果又翻回去 ——
     右边一收、左边看着还是收着的，点了等于没点。 */
  app.classList.toggle('collapsed', willCollapse);
  requestAnimationFrame(paintThink);
});

/* ══════════ 右侧产物栏 ══════════ */
/* 只读地列「产物」文件夹里已经落盘的东西：引擎画完的 SVG、配套提示词、
   写长了落盘的代码。它跟左边那栏互斥 —— 这边一开，左边自动收，反过来也一样。 */
const ART_GLYPH = { img: '◨', text: '≡', other: '◆' };
let artSel = '';          /* 现在选中的是哪一个 */
let artSig = '';          /* 上一轮清单的指纹：一模一样就不重画，免得打断正在看的预览 */
let artTimer = 0;         /* 开着的时候每 6 秒自己扫一遍 */
let artWasCollapsed = false;  /* 开之前左边是收着的吗？关的时候要照原样还回去 */
let artShutTimer = 0;         /* 退场动画的计时器 */
let artShutting = false;      /* 正在往右退 —— 这期间它算"没开" */

function artIsOpen() { const e = $('#art'); return !!e && !e.hidden && !artShutting; }

function artOpen() {
  const e = $('#art');
  if (!e) return;
  /* 还没退干净又点开：把退场掐了，原地复活，别让它闪一下再回来 */
  clearTimeout(artShutTimer); artShutTimer = 0;
  artShutting = false;
  e.classList.remove('art-out');
  const app = $('#app');
  artWasCollapsed = !!app && app.classList.contains('collapsed');
  if (app) app.classList.add('collapsed', 'art-open');   /* 左边让位 */
  e.hidden = false;
  artLoad();
  clearInterval(artTimer);
  artTimer = setInterval(() => {
    if (!artIsOpen()) { clearInterval(artTimer); artTimer = 0; return; }
    artLoad(true);
  }, 6000);
  const c = $('#artClose');
  if (c) setTimeout(() => c.focus(), 60);
}

function artClose() {
  const e = $('#art');
  if (!e || e.hidden || artShutting) return;
  clearInterval(artTimer); artTimer = 0;
  e.classList.remove('art-fresh');
  /* 网格那一步（左边还位、右边收列）马上做，别等动画 ——
     等的话会有 240ms 场上是三列却没东西填，主区先抖一下再归位。 */
  const app = $('#app');
  if (app) {
    app.classList.remove('art-open');
    app.classList.toggle('collapsed', artWasCollapsed);   /* 左边还回它原来的样子 */
  }
  /* 先让它在 .art-out 里滑出去，滑完再真收 */
  artShutting = true;
  e.classList.add('art-out');
  clearTimeout(artShutTimer);
  artShutTimer = setTimeout(() => {
    e.hidden = true;
    e.classList.remove('art-out');
    artShutting = false;
    artShutTimer = 0;
  }, 240);
}

function artSize(n) {
  n = +n || 0;
  if (n < 1024) return n + ' B';
  if (n < 1048576) return (n / 1024).toFixed(n < 10240 ? 1 : 0) + ' KB';
  return (n / 1048576).toFixed(1) + ' MB';
}

function artTime(t) {
  const d = new Date((+t || 0) * 1000), now = new Date();
  const p = x => (x < 10 ? '0' : '') + x;
  const hm = p(d.getHours()) + ':' + p(d.getMinutes());
  if (d.toDateString() === now.toDateString()) return '今天 ' + hm;
  return (d.getMonth() + 1) + '/' + d.getDate() + ' ' + hm;
}

function artUrl(name) { return '/api/artifact/' + encodeURIComponent(String(name)); }

function artKindOf(name) {
  const n = String(name || '').toLowerCase();
  if (/\.(svg|png|jpe?g|webp|gif|bmp|ico)$/.test(n)) return 'img';
  if (/\.(txt|md|py|js|ts|css|html?|json|csv|log|xml|ya?ml|bat|ps1|sh|ini)$/.test(n)) return 'text';
  return 'other';
}

function artViewEmpty() {
  const v = $('#artView');
  if (v) v.innerHTML =
    '<div class="art-empty">左边点一个文件，这里就能看。<br>' +
    '<span>图直接显示，代码和文本原样读给你。</span></div>';
}

function artLoad(quiet) {
  const list = $('#artList');
  if (!list) return Promise.resolve();
  return api('/api/artifacts').then(r => {
    if (!r || !r.ok) { if (!quiet) toast('扫不到产物文件夹'); return; }
    const dirEl = $('#artDir');
    if (dirEl) dirEl.textContent = r.dir || '';
    const items = r.items || [];
    const sig = items.map(it => it.name + ':' + it.mtime + ':' + it.size).join('|');
    /* 后台每 6 秒这一趟：清单没变就什么都别动（不重画 = 不闪、不丢滚动位置） */
    if (quiet && sig === artSig) return;
    const grew = !!artSig && sig !== artSig;
    artSig = sig;
    if (grew) {
      const box = $('#art');
      if (box) box.classList.add('art-fresh');
    }
    if (!items.length) {
      list.innerHTML = '<div class="art-empty">还是空的。<br>' +
        '<span>让它画张图、写段长代码，这里就会冒出来。</span></div>';
      artSel = '';
      artViewEmpty();
      return;
    }
    /* 选中的那个要是被清掉了，预览也跟着清 */
    if (artSel && !items.some(it => it.name === artSel)) { artSel = ''; artViewEmpty(); }
    list.innerHTML = items.map(it =>
      '<button class="art-item' + (it.name === artSel ? ' on' : '') + '" data-n="' +
        esc(it.name) + '" title="' + esc(it.name) + '">' +
        '<i class="ai-kind">' + (ART_GLYPH[it.kind] || '◆') + '</i>' +
        '<span class="ai-tx"><b>' + esc(it.name) + '</b>' +
        '<em>' + artSize(it.size) + ' · ' + artTime(it.mtime) + '</em></span>' +
      '</button>').join('');
  });
}

function artPick(name) {
  const v = $('#artView');
  if (!v) return;
  const box = $('#art');
  if (box) box.classList.remove('art-fresh');
  artSel = name;
  $$('#artList .art-item').forEach(b => b.classList.toggle('on', b.dataset.n === name));

  const url = artUrl(name);
  const head = '<div class="art-meta"><span>' + esc(name) + '</span>' +
               '<a href="' + url + '" target="_blank" rel="noopener">打开</a>' +
               '<a href="' + url + '?dl=1">下载</a></div>';
  const kind = artKindOf(name);

  if (kind === 'img') {
    v.innerHTML = head + '<img src="' + url + '" alt="' + esc(name) + '">';
    return;
  }
  if (kind === 'other') {
    v.innerHTML = head + '<div class="art-note">这种格式没法在这儿显示。<br>' +
                         '点上面的「打开」或「下载」把它拿走。</div>';
    return;
  }
  v.innerHTML = head + '<div class="art-note">正在读…</div>';
  fetch(url).then(r => r.text()).then(t => {
    if (artSel !== name) return;
    const cut = t.length > 40000 ? t.slice(0, 40000) + '\n\n…（后面还有，点「打开」看全的）' : t;
    v.innerHTML = head + '<pre></pre>';
    const pre = v.querySelector('pre');
    if (pre) pre.textContent = cut;   /* textContent：原样贴，别让代码被当成 HTML */
  }).catch(() => {
    if (artSel === name) v.innerHTML = head + '<div class="art-note">读不出来，点「打开」试试。</div>';
  });
}

const artListEl = $('#artList');
if (artListEl) artListEl.addEventListener('click', ev => {
  const b = ev.target.closest('.art-item');
  if (b && b.dataset.n) artPick(b.dataset.n);
});
const artRefreshBtn = $('#artRefresh');
if (artRefreshBtn) artRefreshBtn.addEventListener('click', () => { artSig = ''; artLoad(); });
const artCloseBtn = $('#artClose');
if (artCloseBtn) artCloseBtn.addEventListener('click', artClose);
const artBtnEl = $('#artBtn');
if (artBtnEl) artBtnEl.addEventListener('click', () => { if (artIsOpen()) artClose(); else artOpen(); });

/* ══════════ 输入框里那个会自己打字的提示 ══════════ */
const HINTS = [
  '想到什么，直接往下说。',
  '试试：帮我把这段话改得顺一点',
  '试试：用三句话讲清楚注意力机制',
  '试试：今天有点累，随便聊聊',
  'Enter 发送 · Shift+Enter 换行',
  '我不联网，所有东西都在这台机器里。'
];
let hintTimer = 0, hintIdx = 0, hintOn = true;

function hintStop() { hintOn = false; clearTimeout(hintTimer); }
function hintStart() {
  if (hintOn) return;
  hintOn = true;
  clearTimeout(hintTimer);
  hintTimer = setTimeout(hintLoop, 900);
}
function hintLoop() {
  if (!hintOn) return;
  if (REDUCED) { input.placeholder = HINTS[hintIdx % HINTS.length]; hintIdx++;
                 hintTimer = setTimeout(hintLoop, 4200); return; }
  const s = HINTS[hintIdx % HINTS.length];
  const chars = Array.from(s);
  let i = 0;
  const type = () => {
    if (!hintOn) { input.placeholder = ''; return; }
    i++;
    input.placeholder = chars.slice(0, i).join('') + (i < chars.length ? '▍' : '');
    if (i < chars.length) {
      hintTimer = setTimeout(type, Math.max(26, 62 * rnd(0.62, 1.45)));
    } else {
      hintTimer = setTimeout(() => {
        const back = () => {
          if (!hintOn) return;
          i -= 2;
          if (i > 0) { input.placeholder = chars.slice(0, i).join('') + '▍';
                       hintTimer = setTimeout(back, 22); }
          else { input.placeholder = ''; hintIdx++;
                 hintTimer = setTimeout(hintLoop, 420); }
        };
        back();
      }, 1900);
    }
  };
  hintTimer = setTimeout(type, 240);
}

/* ══════════ 铭牌 · 机器信息卡 ══════════ */
function pickSteps(train, tier) {
  if (!train || !train.length) return null;
  const t = String(tier || '');
  const hit = t ? train.find(x => x.file && x.file.indexOf(t) >= 0) : null;
  const it = hit || (train.length === 1 ? train[0] : null);
  return (it && it.steps != null) ? it.steps : null;
}

async function refreshPlate() {
  const a = await api('/api/about');
  if (!a || a.ok === false) {
    S.plate = null;
    $('#plateSn').textContent = 'SN —';
    return;
  }
  S.plate = a;
  const steps = pickSteps(a.train, a.tier);
  S.trainSteps = steps;
  $('#plateSn').textContent = 'SN ' + (steps == null ? '—' : steps);
  const seal = $('#plateSeal');
  const official = !!a.model_id;
  seal.textContent = official ? '正式版' : (a.legacy ? '老版本' : '其他版本');
  seal.classList.toggle('gray', !official);
}

function openSheet() {
  const wrap = $('#sheet'), box = $('#sheetCard');
  wrap.hidden = false;
  box.innerHTML = '<h3>小方</h3><div class="sh-sub">正在读这台机器…</div>';

  api('/api/about').then(a => {
    if (!a || a.ok === false) {
      box.innerHTML = '<h3>小方</h3><div class="sh-sub">读不到</div>' +
        '<div class="sh-sec"><h5>说明</h5><div class="sh-npz"><div>' +
        '<span>' + (a && a.net ? '口子还没通' : '这个版本的桥接程序还没带信息接口') +
        '</span><span>—</span></div></div></div>' +
        '<button class="sh-close" id="shClose">知道了</button>';
      bindSheetClose(box);
      return;
    }

    const steps = pickSteps(a.train, a.tier);
    const official = !!a.model_id;
    const stTxt = STATE_TXT[a.state] ? STATE_TXT[a.state][0] : (a.state || '—');
    const rows = [
      ['模型', a.model || '—'],
      ['引擎', a.engine || '—'],
      ['档位', a.tier ? S.tierCn(a.tier) : '—'],
      ['状态', stTxt],
      ['本地缓存', a.cache ? (a.cache.size + ' · ' + a.cache.files + ' 片') : '—'],
      ['历史对话', a.chats == null ? '—' : a.chats + ' 条']
    ];
    let html = '<h3>小方 · 这台机器</h3><div class="sh-sub">' +
      (official ? '正式版' : (a.legacy ? '老版本' : '其他版本')) +
      (a.tier ? ' · ' + String(a.tier).toUpperCase() : '') +
      ' · SN ' + (steps == null ? '—' : steps) + '</div>';
    html += '<dl class="sh-grid">';
    rows.forEach(r => { html += '<dt>' + esc(r[0]) + '</dt><dd>' + esc(r[1]) + '</dd>'; });
    html += '</dl>';

    if (a.train && a.train.length) {
      const t0 = String(a.tier || '');
      html += '<div class="sh-sec"><h5>训练状态（本地那几个 npz）</h5><div class="sh-npz">';
      a.train.forEach(t => {
        const mine = t0 && t.file && t.file.indexOf(t0) >= 0;
        html += '<div><span>' + esc(t.file) + (mine ? ' ← 正在用' : '') + '</span><span class="' +
          (mine ? 'hi' : '') + '">' + esc(t.size || '—') + ' · ' +
          (t.steps == null ? '—' : t.steps + ' 步') +
          (t.last_loss == null ? '' : ' · loss ' + t.last_loss) + '</span></div>';
      });
      html += '</div><div class="sh-npz" style="margin-top:6px">' +
        '<div><span>最近一次改动</span><span>' +
        esc((a.train[0] && a.train[0].mtime) || '—') + '</span></div></div></div>';
    }

    if (a.engine_path) {
      html += '<div class="sh-sec"><h5>引擎文件</h5><div class="sh-npz"><div>' +
        '<span>' + esc(a.engine_path) + '</span><span>—</span></div></div></div>';
    }

    html += '<div class="sh-sec"><h5>一句话</h5><div class="sh-npz"><div>' +
      '<span>全在本机跑，不出网；关掉窗口，推理进程跟着收摊</span><span>—</span></div>' +
      '</div></div><button class="sh-close" id="shClose">知道了</button>';
    box.innerHTML = html;
    bindSheetClose(box);
  });
}

function bindSheetClose(box) {
  const b = box.querySelector('#shClose');
  if (b) b.addEventListener('click', closeSheet);
}
function closeSheet() { $('#sheet').hidden = true; }
$('#plate').addEventListener('click', e => {
  e.stopPropagation();
  if ($('#sheet').hidden) openSheet(); else closeSheet();
});
$('#sheet').addEventListener('click', e => {
  if (e.target.id === 'sheet' || !e.target.closest('.sheet-card')) closeSheet();
});

/* ══════════ 首屏视差 ══════════ */
function initParallax() {
  const wrap = $('#ghostWrap'), card = $('#helloCard'), ghost = $('#helloGhost');
  if (!wrap || REDUCED) return;

  /* 拍一下那只小方，它应你一声 */
  wrap.classList.add('playable');
  wrap.addEventListener('pointerenter', () => ghost && ghost.classList.add('peek'));
  wrap.addEventListener('pointerleave', () => ghost && ghost.classList.remove('peek'));
  wrap.addEventListener('click', () => {
    if ($('#hello').hidden) return;
    wrap.classList.remove('tap'); void wrap.offsetWidth; wrap.classList.add('tap');
    setTimeout(() => wrap.classList.remove('tap'), 1300);
    const lines = ['我在呢，说吧。', '想到什么就打下面那行。', '不用客气，直说就行。',
                   '手边这些都能试。'];
    toast(lines[Math.floor(Math.random() * lines.length)]);
    input.focus();
  });

  if (card) card.addEventListener('animationend', () => { card.style.animation = 'none'; });

  let tx = 0, ty = 0, cx = 0, cy = 0, raf = 0, live = false;
  const tick = () => {
    cx += (tx - cx) * 0.085;
    cy += (ty - cy) * 0.085;
    /* 背景那只：左右上下漂、绕两个轴轻微转身，再顺着指针方向往镜头里推。
       平移给"远近"，旋转给"立体"，两者都只碰 transform，不触发重排。
       旋转刻意压得很轻（±7° 上下），不然读字的人会晕。 */
    const gx = (-cx * 24).toFixed(2), gy = (-cy * 18).toFixed(2);
    const gz = (12 + Math.abs(cx) * 46 + Math.abs(cy) * 32).toFixed(2);
    wrap.style.transform = 'translate3d(' + gx + 'px,' + gy + 'px,' + gz + 'px) rotateY(' +
      (-cx * 7.5).toFixed(2) + 'deg) rotateX(' + (cy * 5.5).toFixed(2) +
      'deg) rotateZ(' + (cx * 1.3).toFixed(2) + 'deg)';
    if (card) {
      /* 卡片反着来：往指针同侧挪一点，同时被推远。
         一个近一个远，两层之间就撑出真实的空间距离。 */
      card.style.transform = 'translate3d(' + (cx * 6).toFixed(2) + 'px,' +
        (cy * 4.5).toFixed(2) + 'px,' + (-Math.abs(cx) * 26).toFixed(2) +
        'px) rotateY(' + (cx * 2.2).toFixed(2) +
        'deg) rotateX(' + (-cy * 1.6).toFixed(2) + 'deg)';
    }
    if (Math.abs(tx - cx) > 0.0015 || Math.abs(ty - cy) > 0.0015) {
      raf = requestAnimationFrame(tick);
    } else { live = false; }
  };
  window.addEventListener('pointermove', e => {
    if ($('#hello').hidden) return;
    tx = (e.clientX / window.innerWidth - 0.5) * 2;
    ty = (e.clientY / window.innerHeight - 0.5) * 2;
    if (!live) { live = true; raf = requestAnimationFrame(tick); }
  }, { passive: true });
}

/* ── 事件流 ─────────────────────────────────────────────────────────── */
/* 兜底用的惰性计时器。真正"答完了"由引擎的 idle 事件负责。以前这里是 1.5 秒，
   长思考时两段 think 之间常隔好几秒，网页会在回答中途把"思考中"闪成"就绪"，
   所以拉到 45 秒，只当最后一根稻草。 */
function nudgeIdle() {
  clearTimeout(S.idleTimer);
  S.idleTimer = setTimeout(() => {
    if (S.state === 'working') setState('ready');
  }, 45000);
}

function onEvent(ev) {
  const t = ev.t, x = ev.x || '';
  switch (t) {
    case 'state':
      setState(x, ev.detail);
      if (x === 'loading') ovLog(ev.detail || '正在把模型搬进内存…');
      break;

    case 'ready':
      setState('ready', x);
      if (x) $('#curModelSub').textContent = x;
      refreshPlate();
      break;

    case 'model': {
      S.model = x; S.modelId = ev.model_id || S.modelId;
      S.tier = ev.tier || ''; S.legacy = !!ev.legacy;
      $('#curModel').textContent = x;
      $('#mbName').textContent = x;
      if (S.flat[S.modelId]) {
        $('#curModelSub').textContent = S.flat[S.modelId].desc || '';
        $('#mbDot').className = 'mb-dot' + (S.flat[S.modelId].ready ? '' : ' no');
      } else {
        $('#curModelSub').textContent = '本地引擎 · ' + (ev.engine || '');
      }
      refreshConvs();
      refreshPlate();
      break;
    }

    case 'user':
      if (x === S.lastUser) { S.lastUser = ''; }
      else { bubbleRow('me', x); }
      setState('working');
      break;

    case 'think':
      thinkRow(x);
      nudgeIdle();
      break;

    case 'say':
      bubbleRow('ai', x);
      nudgeIdle();
      refreshConvs();
      break;

    case 'idle':
      clearTimeout(S.idleTimer);
      if (S.state === 'working') setState('ready');
      break;

    case 'cmd':
      sysRow(x, 'cmd');
      break;

    case 'log':
      if (x.indexOf('[') === 0) sysRow(x);
      else if (!(S.legacy && /^(你|小方)\s*[:：]/.test(x))) runLogRow(x);
      ovLog(x);
      break;

    case 'conv':
      S.cid = x;
      S.convName = ev.name || '';
      if (ev.msgs) {
        msgs.innerHTML = ''; S.runCount = 0;
        ev.msgs.forEach(m => bubbleRow(m.role === 'user' ? 'me' : 'ai', m.text,
                                      { instant: true }));
        if (ev.msgs.length) hideHello(); else showHello(true);
        S.nearBottom = true;
        stream.scrollTop = stream.scrollHeight;
      }
      refreshConvs();
      break;

    case 'bye':
      setState('down', x);
      break;

    case 'loading':
      ovTitle(x);
      showOverlay();
      break;
  }
}

function connect() {
  if (S.es) { try { S.es.close(); } catch (e) {} }
  const es = new EventSource('/api/stream?since=' + S.since);
  S.es = es;
  S.lastEvt = Date.now();

  es.onmessage = e => {
    let ev = null;
    try { ev = JSON.parse(e.data); } catch (err) { return; }
    S.lastEvt = Date.now();
    if (ev.i) S.since = Math.max(S.since, ev.i);
    onEvent(ev);
  };
  es.addEventListener('bye', () => setState('down', '引擎退出了'));
  es.onerror = () => {
    if (S.state !== 'down') $('#connTxt').textContent = '连接断了，正在重连…';
  };

  /* 有些情况下 SSE 会"半死"：不发错也不发东西。自己掐表重连。 */
  clearInterval(S.esWatch);
  S.esWatch = setInterval(() => {
    if (S.state === 'down' || !S.booted) return;
    if (Date.now() - S.lastEvt > 24000) { S.lastEvt = Date.now(); connect(); }
  }, 5000);
}

/* ── 心跳：网页不吭声 10 秒，服务端就认为窗口关了，收摊 ─────────────── */
function startBeat() {
  const beat = () => fetch('/api/beat', { method: 'POST' }).catch(() => {});
  beat();
  setInterval(beat, S.beatMs);
}
window.addEventListener('pagehide', () => {
  try { navigator.sendBeacon('/api/bye'); } catch (e) {}
});

/* ── 连不上时的兜底页 ───────────────────────────────────────────────── */
function showBoot(tries, r) {
  const b = $('#boot');
  b.hidden = false;
  $('#bootTitle').innerHTML = '正在唤醒小方<span class="dots"></span>';
  $('#bootSub').textContent = (r && r.net)
    ? '本地那个口子还没开，网页在一下一下地敲（第 ' + tries + ' 次）。'
    : '桥接程序还没准备好，网页在重试（第 ' + tries + ' 次）。';
  const u = $('#bootUrl');
  u.hidden = false;
  u.textContent = location.href;
  $('#bootTips').innerHTML =
    '<b>1.</b> 那个最小化的黑色命令窗口，是小方的推理进程 —— 别关它。<br>' +
    '<b>2.</b> 一直连不上，就回去把网页那一步重新启动一次。<br>' +
    '<b>3.</b> 上面这个地址就是它该在的地方；端口要是换过，以那个窗口里打印的为准。';
}

/* 老版本的桥接 py 不一定有 /api/boot、/api/about，缺了也得能进来 */
async function ensureServer() {
  for (let i = 1; i <= 220; i++) {
    const r = await api('/api/boot');
    if (r && r.ok) return r;
    if (r && !r.net && (r.http === 404 || r.http === 500)) {
      return { ok: true, degraded: true, state: 'starting', model: '', model_id: '',
               tier: '', engine: '', legacy: false, catalog: [] };
    }
    showBoot(i, r);
    await sleep(i < 6 ? 400 : 850);
  }
  return null;
}

/* ── 起步 ───────────────────────────────────────────────────────────── */
async function boot() {
  const b = await ensureServer();
  if (!b) return;
  $('#boot').hidden = true;
  S.booted = true;
  S.degraded = !!b.degraded;

  S.catalog = b.catalog || [];
  S.flat = {};
  (function walk(ns) { (ns || []).forEach(n => {
    if (n.engine) S.flat[n.id] = n;
    walk(n.children); walk(n.items);
  }); })(S.catalog);

  S.modelId = b.model_id || ''; S.model = b.model || '';
  S.tier = b.tier || ''; S.legacy = !!b.legacy;
  S.since = b.since || 0;
  S.beatMs = b.beat || 1500;
  S.hold = b.hold || 10;
  S.cid = (b.conv && b.conv.cid) || '';
  S.convName = (b.conv && b.conv.name) || '';
  S.think.mode = (b.think && b.think.mode) || 'think';
  S.think.turns = (b.think && b.think.turns) || 3;
  /* 八项设置的真相在 xiaofang_settings.py 里，boot 时一次性捞回来；
     老版本的后端不吐 settings，那就退回 think.level 这一项。 */
  if (b.settings) Object.assign(S.settings, b.settings);
  else if (b.think && b.think.level !== undefined) S.settings.FLASH_LEVEL = b.think.level;
  cfgPaint();

  $('#curModel').textContent = b.model || 'Covi1 Lite';
  $('#mbName').textContent = b.model || 'Covi1 Lite';
  if (S.flat[S.modelId]) {
    $('#curModelSub').textContent = S.flat[S.modelId].desc || '';
    $('#mbDot').className = 'mb-dot' + (S.flat[S.modelId].ready ? '' : ' no');
  } else {
    $('#curModelSub').textContent = b.engine
      ? ('本地引擎 · ' + b.engine) : '本地引擎';
  }
  $('#modelBtn').hidden = !S.catalog.length;
  $('#dockHint').innerHTML = S.legacy
    ? '这一档是老命令行引擎，网页只把它说的话原样搬过来。<b>别关那个黑色小窗</b>，关了它小方就跟着退。'
    : '模型跑在一个最小化的黑色命令窗口里，那是我本地的推理进程，<b>别关它</b>。';

  paintThink();
  autoGrow();
  setState(b.state || 'starting');
  showHello(false);
  refreshConvs();
  refreshPlate();
  connect();
  startBeat();
  initParallax();
  hintStart();
  setTimeout(() => input.focus(), 200);
}

/* ══════════ 背景图：列表 · 大预览 · 两片交叉淡入 ══════════ */
/* 两片 .bg-img 一直叠着不动，换图只是新的那片透明度升上去、旧的压下来 ——
   全程只有 opacity 在动，不重排、也不重绘一大块 */
const BG_KEY = 'xf.bg';
const bgWrap = $('#bgWrap');
const bgA = $('#bgImgA');
const bgB = $('#bgImgB');
let bgFront = 0;      /* 现在露在外面的是哪一片：0 是 A，1 是 B */
let bgCur = '';       /* 当前背景地址，空串就是用默认的浅色底 */
let bgFiles = [];     /* 文件夹里有哪些图 */

function bgApply(url, animate) {
  if (!bgWrap || !bgA || !bgB) return;
  bgCur = url || '';

  if (!url) {
    /* 恢复默认：两片一起淡掉，底下的浅色底自己就回来了 */
    bgA.classList.remove('on');
    bgB.classList.remove('on');
    bgWrap.classList.remove('has');
    bgFront = 0;
    return;
  }

  const next = bgFront === 0 ? bgB : bgA;
  const prev = bgFront === 0 ? bgA : bgB;
  next.style.backgroundImage = 'url("' + String(url).replace(/["\\\r\n]/g, '') + '")';

  if (!animate || REDUCED) {
    bgA.classList.remove('on');
    bgB.classList.remove('on');
    next.classList.add('on');
  } else {
    /* 先让新的那片按全透明算一帧，再翻透明度 ——
       同一帧里又改 style 又改 class，过渡有可能被浏览器吞掉 */
    void next.offsetWidth;
    next.classList.add('on');
    prev.classList.remove('on');
  }
  bgWrap.classList.add('has');
  bgFront = next === bgA ? 0 : 1;
}

function bgSave(url) {
  bgCur = url || '';
  try {
    if (url) localStorage.setItem(BG_KEY, url);
    else localStorage.removeItem(BG_KEY);
  } catch (_) {}
  bgMark();
}

function bgRestore() {
  let url = '';
  try { url = localStorage.getItem(BG_KEY) || ''; } catch (_) {}
  if (url) bgApply(url, false);
}

/* ── 缩略图墙 ───────────────────────────────────────────────────────── */
function bgMark() {
  $$('#bgGrid .sk-th').forEach(b => b.classList.toggle('on', b.dataset.url === bgCur));
}

function bgPaint() {
  const grid = $('#bgGrid');
  if (!grid) return;
  const cnt = $('#bgCount'), empty = $('#bgEmpty');
  grid.innerHTML = '';
  if (cnt) cnt.textContent = String(bgFiles.length);
  if (empty) {
    empty.hidden = bgFiles.length > 0;
    empty.textContent = '文件夹里还空着。把 PNG / JPG / WEBP 拖进上面那块，图就会出现在这儿。';
  }
  bgFiles.forEach((f, i) => {
    const th = document.createElement('button');
    th.type = 'button';
    th.className = 'sk-th';
    th.dataset.url = f.url;
    th.title = f.name;
    th.style.animationDelay = Math.min(i * 26, 340) + 'ms';
    const im = document.createElement('img');
    im.src = f.url; im.alt = f.name; im.loading = 'lazy'; im.decoding = 'async';
    const nm = document.createElement('span');
    nm.className = 'sk-th-nm'; nm.textContent = f.name;
    th.appendChild(im); th.appendChild(nm);
    grid.appendChild(th);
  });
  bgMark();
}

/* ── 问后端要列表 ───────────────────────────────────────────────────── */
async function bgLoad(quiet) {
  const grid = $('#bgGrid');
  if (!grid) return;
  grid.setAttribute('aria-busy', 'true');
  const r = await api('/api/backgrounds');
  grid.removeAttribute('aria-busy');

  const empty = $('#bgEmpty');
  if (!r || !r.ok) {
    bgFiles = [];
    bgPaint();
    if (empty) {
      empty.hidden = false;
      empty.textContent = (r && r.net)
        ? '没连上本地那个口子，暂时看不到文件夹里的图。'
        : '这个版本的后端还没有背景图接口，先手动把图放进「背景图」文件夹也一样。';
    }
    if (!quiet) toast('背景图列表没取到');
    return;
  }

  const dir = $('#bgDir');
  if (dir && r.dir) dir.textContent = r.dir;
  bgFiles = (r.files || [])
    .filter(f => f && f.url)
    .map(f => ({ name: f.name || String(f.url).split('/').pop(), url: f.url }));
  bgPaint();
  if (!quiet) toast('背景图 · 一共 ' + bgFiles.length + ' 张');
}

/* ── 悬停时跟出来的大预览 ───────────────────────────────────────────── */
const bgPeek = $('#bgPeek');
let peekUrl = '', peekTimer = 0;

/* 位置写进 --px/--py，动的还是 transform；贴边就翻到另一边 */
function peekAt(ev) {
  if (!bgPeek) return;
  const w = 246, h = 154, m = 14;
  let x = ev.clientX + 22;
  let y = ev.clientY - h / 2;
  if (x + w + m > window.innerWidth) x = ev.clientX - w - 22;
  x = Math.max(m, Math.min(window.innerWidth - w - m, x));
  y = Math.max(m, Math.min(window.innerHeight - h - m, y));
  bgPeek.style.setProperty('--px', Math.round(x) + 'px');
  bgPeek.style.setProperty('--py', Math.round(y) + 'px');
}

function peekShow(th, ev) {
  if (!bgPeek) return;
  clearTimeout(peekTimer);
  const url = th.dataset.url;
  if (peekUrl !== url) {
    peekUrl = url;
    bgPeek.innerHTML = '<img src="' + esc(url) + '" alt=""><b>' + esc(th.title || '') + '</b>';
  }
  peekAt(ev);
  bgPeek.hidden = false;
  requestAnimationFrame(() => { if (bgPeek) bgPeek.classList.add('show'); });
}

function peekHide() {
  if (!bgPeek) return;
  clearTimeout(peekTimer);
  bgPeek.classList.remove('show');
  /* 等淡出跑完再收起来，不然那 0.2 秒的渐隐根本看不见 */
  peekTimer = setTimeout(() => {
    bgPeek.hidden = true;
    bgPeek.innerHTML = '';
    peekUrl = '';
  }, 220);
}

const bgGrid = $('#bgGrid');
if (bgGrid) {
  bgGrid.addEventListener('pointerover', ev => {
    const th = ev.target.closest('.sk-th');
    if (th) peekShow(th, ev);
  });
  bgGrid.addEventListener('pointermove', ev => {
    if (!bgPeek || bgPeek.hidden) return;
    const th = ev.target.closest('.sk-th');
    if (th) peekAt(ev); else peekHide();
  });
  bgGrid.addEventListener('pointerout', ev => {
    const th = ev.target.closest('.sk-th');
    if (!th) return;
    if (ev.relatedTarget && th.contains(ev.relatedTarget)) return;
    peekHide();
  });
  bgGrid.addEventListener('scroll', peekHide, { passive: true });
  /* 点一下就是它了 */
  bgGrid.addEventListener('click', ev => {
    const th = ev.target.closest('.sk-th');
    if (!th) return;
    bgApply(th.dataset.url, true);
    bgSave(th.dataset.url);
    peekHide();
    toast('背景换成 · ' + (th.title || '这张'));
  });
}

/* ── 开合面板 ───────────────────────────────────────────────────────── */
function skinIsOpen() { const sk = $('#skin'); return !!sk && !sk.hidden; }

function openSkin() {
  const sk = $('#skin');
  if (!sk) return;
  const c = $('#cfg');
  if (c && !c.hidden) closeCfg();          /* 两块浮层不同时杵着 */
  sk.hidden = false;
  bgLoad(true);
  const cc = $('#skinClose');
  if (cc) setTimeout(() => cc.focus(), 60);
}

function closeSkin() {
  const sk = $('#skin');
  if (!sk || sk.hidden) return;
  sk.hidden = true;
  peekHide();
  const b = $('#skinBtn');
  if (b) b.focus();
}

const skinBtn = $('#skinBtn');
if (skinBtn) skinBtn.addEventListener('click', () => {
  if (skinIsOpen()) closeSkin(); else openSkin();
});

const skinEl = $('#skin');
if (skinEl) skinEl.addEventListener('pointerdown', ev => {
  if (ev.target === skinEl) closeSkin();     /* 点卡片外头就收 */
});

const skinCloseBtn = $('#skinClose');
if (skinCloseBtn) skinCloseBtn.addEventListener('click', closeSkin);

const bgRefreshBtn = $('#bgRefresh');
if (bgRefreshBtn) bgRefreshBtn.addEventListener('click', () => bgLoad(false));

const bgDefaultBtn = $('#bgDefault');
if (bgDefaultBtn) bgDefaultBtn.addEventListener('click', () => {
  if (!bgCur) { toast('本来就是默认那张底'); return; }
  bgApply('', true);
  bgSave('');
  toast('背景恢复默认');
});

/* ══════════ 设置面板 ══════════ */
/* 面板上这八项跟命令行里敲 `setting` 改的是同一批开关、同一个文件 ——
   网页这边改完立刻写盘，顺手把等价的 /setting 喂给引擎做热更新，
   所以不必去关那个黑色小窗重启。字段名跟 xiaofang_settings.py 一字不差。 */
const CF_DEF = {
  THINK_MODE: 'think', THINK_TURNS: 3,
  EMOTION_SENSITIVITY: 1.2, ENERGY: 0.9,
  USE_PUNCT_EMOJI: true, SHOW_DEEP_THINK: true,
  FORCE_OFFLINE: false, FLASH_LEVEL: 2
};
const CF_SPEED = ['原速', '轻快', 'Flash', '极速'];
const CF_MODE_LONG = { off: '不思考', think: '单轮思考', multi: '多轮边想' };

/* boot() 捞回来的那份会盖在这上面；往后所有改动都以 S.settings 为准 */
S.settings = Object.assign({}, CF_DEF);

function cfgIsOpen() { const e = $('#cfg'); return !!e && !e.hidden; }

function openCfg() {
  const e = $('#cfg');
  if (!e) return;
  const sk = $('#skin');
  if (sk && !sk.hidden) closeSkin();       /* 两块浮层不同时杵着 */
  e.hidden = false;
  cfgPaint();
  /* 每次开都从盘上捞一遍：命令行那边可能刚改过 */
  api('/api/settings').then(r => {
    if (r && r.ok && r.settings) { Object.assign(S.settings, r.settings); cfgPaint(); }
  });
  const c = $('#cfgClose');
  if (c) setTimeout(() => c.focus(), 60);
}

function closeCfg() {
  const e = $('#cfg');
  if (!e || e.hidden) return;
  e.hidden = true;
  const b = $('#cfgBtn');
  if (b) b.focus();
}

/* ── 滑块：圆球 + 跟着球走的角标 ─────────────────────────────────── */
/* 只往 CSS 写一个 0~1 的比例（--r），像素全交给 calc 去算 ——
   跟首屏那根三档滑块「--fillR 配 scaleX」是同一套思路，JS 不碰尺寸，
   所以窗口缩放、换皮都不用重算。 */
const CF_KNOB = 18;                    /* 必须跟 style.css 里 .cf-knob 的宽高一致 */
const CF_INSET = (26 - CF_KNOB) / 2;   /* 轨道高 26、球 18 → 上下各让 4 */

function cfBuildSlide(el) {
  if (!el || el.classList.contains('built')) return;
  const row = el.closest('.cf-row');
  const lab = row && row.querySelector('.cf-lab b');
  el.classList.add('built');
  el.innerHTML =
    '<div class="cf-track" role="slider" tabindex="0" aria-label="' +
      esc(lab ? lab.textContent : '设置') + '" aria-valuemin="' + el.dataset.min +
      '" aria-valuemax="' + el.dataset.max + '">' +
      '<div class="cf-fill"></div><div class="cf-knob"></div>' +
    '</div><div class="cf-bub"></div>';
  el._trk = el.querySelector('.cf-track');
  el._bub = el.querySelector('.cf-bub');
}

/* 落到步长上：1 / 0.1 两种刻度都走这一条，顺带把浮点毛刺抹平 */
function cfSnap(el, v) {
  const lo = +el.dataset.min, hi = +el.dataset.max, st = +el.dataset.step;
  v = Math.max(lo, Math.min(hi, v));
  v = lo + Math.round((v - lo) / st) * st;
  const dec = (String(st).split('.')[1] || '').length;
  return +v.toFixed(dec + 1);
}

function cfFmt(el, v) {
  return el.dataset.mode === 'int' ? String(Math.round(v)) : (+v).toFixed(1);
}

function cfPaintSlide(el, v) {
  if (!el) return;
  cfBuildSlide(el);
  const lo = +el.dataset.min, hi = +el.dataset.max, k = hi > lo ? (v - lo) / (hi - lo) : 0;
  el.style.setProperty('--r', Math.max(0, Math.min(1, k)).toFixed(4));
  if (el._bub) el._bub.textContent = cfFmt(el, v);
  if (el._trk) el._trk.setAttribute('aria-valuenow', String(v));
}

/* 指针横坐标 → 值。球心能走的那一段是 [INSET+半球, 宽-INSET-半球]，
   跟 CSS 里 `left:calc(4px + var(--r)*(100% - 26px))` 是同一个坐标系。 */
function cfAt(el, clientX) {
  const lo = +el.dataset.min, hi = +el.dataset.max;
  const inner = el._trk.clientWidth - CF_KNOB - CF_INSET * 2;
  if (inner <= 0) return lo;
  const x = clientX - el._trk.getBoundingClientRect().left - CF_INSET - CF_KNOB / 2;
  return cfSnap(el, lo + Math.max(0, Math.min(1, x / inner)) * (hi - lo));
}

function cfWireSlide(el, key) {
  if (!el) return;
  cfBuildSlide(el);
  let drag = null;

  const put = clientX => {
    const v = cfAt(el, clientX);
    S.settings[key] = v;
    cfPaintSlide(el, v);
    cfCurLine();
  };

  el._trk.addEventListener('pointerdown', ev => {
    drag = ev.pointerId;
    el.classList.add('drag');
    try { el._trk.setPointerCapture(ev.pointerId); } catch (_) {}
    put(ev.clientX);
    ev.preventDefault();
  });
  el._trk.addEventListener('pointermove', ev => {
    if (drag === null || ev.pointerId !== drag) return;
    put(ev.clientX);
  });
  const drop = ev => {
    if (drag === null || ev.pointerId !== drag) return;
    drag = null;
    el.classList.remove('drag');
    try { el._trk.releasePointerCapture(ev.pointerId); } catch (_) {}
    cfPush(key, S.settings[key]);
  };
  el._trk.addEventListener('pointerup', drop);
  el._trk.addEventListener('pointercancel', drop);

  el._trk.addEventListener('keydown', ev => {
    const lo = +el.dataset.min, hi = +el.dataset.max, st = +el.dataset.step;
    let v = +S.settings[key];
    if (ev.key === 'ArrowRight' || ev.key === 'ArrowUp') v += st;
    else if (ev.key === 'ArrowLeft' || ev.key === 'ArrowDown') v -= st;
    else if (ev.key === 'Home') v = lo;
    else if (ev.key === 'End') v = hi;
    else return;
    ev.preventDefault();
    v = cfSnap(el, v);
    S.settings[key] = v;
    cfPaintSlide(el, v);
    cfCurLine();
    cfPush(key, v);
  });
}

/* ── 开关的三件套 ─────────────────────────────────────────────────── */
function cfSwitch(el, on) {
  if (!el) return;
  el.classList.toggle('on', !!on);
  el.setAttribute('aria-checked', on ? 'true' : 'false');
}

/* 顶上那行摘要：一眼看清现在是什么配置 */
function cfCurLine() {
  const el = $('#cfCur'), st = S.settings;
  if (!el) return;
  const yn = v => (v ? '开' : '关');
  el.innerHTML =
    '思考 <b>' + (CF_MODE_LONG[st.THINK_MODE] || st.THINK_MODE) +
      (st.THINK_MODE === 'multi' ? ' ' + st.THINK_TURNS + ' 轮' : '') + '</b>' +
    ' · 情绪 <b>' + (+st.EMOTION_SENSITIVITY).toFixed(1) + '</b>' +
    ' · 活泼 <b>' + (+st.ENERGY).toFixed(1) + '</b>' +
    ' · 速度 <b>' + (CF_SPEED[st.FLASH_LEVEL] || 'Flash') + '</b>' +
    ' · 标点 <b>' + yn(st.USE_PUNCT_EMOJI) + '</b>' +
    ' · 深度 <b>' + yn(st.SHOW_DEEP_THINK) + '</b>' +
    ' · 离线 <b>' + yn(st.FORCE_OFFLINE) + '</b>';
}

function cfgPaint() {
  const st = S.settings;
  $$('#cfMode button').forEach(b => b.classList.toggle('on', b.dataset.v === st.THINK_MODE));
  $$('#cfSpeed button').forEach(b => b.classList.toggle('on', +b.dataset.v === +st.FLASH_LEVEL));
  cfPaintSlide($('#cfTurns'), st.THINK_TURNS);
  cfPaintSlide($('#cfEmo'), st.EMOTION_SENSITIVITY);
  cfPaintSlide($('#cfEnergy'), st.ENERGY);
  cfSwitch($('#cfPunct'), st.USE_PUNCT_EMOJI);
  cfSwitch($('#cfDeep'), st.SHOW_DEEP_THINK);
  cfSwitch($('#cfNet'), st.FORCE_OFFLINE);
  cfCurLine();
}

/* 单一出口：先本地记下（界面马上有反应），再跟后端对账。
   思考模式 / 轮数这两项走首屏那根滑块的老接口 —— 引擎那边认的是
   /off、/think、/multi N 这三句，比通用接口更准。 */
function cfPush(key, val) {
  const card = $('#cfgCard'), patch = {};
  patch[key] = val;
  let req;
  if (key === 'THINK_MODE' || key === 'THINK_TURNS') {
    if (key === 'THINK_MODE') S.think.mode = val; else S.think.turns = val;
    paintThink();
    req = api('/api/think', { mode: S.think.mode, turns: S.think.turns });
  } else {
    req = api('/api/settings', patch);
  }
  if (card) card.classList.add('busy');
  return req.then(r => {
    if (card) card.classList.remove('busy');
    if (!r || !r.ok) toast('这项没写进去，回头再试一次');
    /* 不管成没成，都回盘上核一遍 —— 界面显示的就是文件里真实的值 */
    return api('/api/settings').then(q => {
      if (q && q.ok && q.settings) { Object.assign(S.settings, q.settings); cfgPaint(); }
      return r;
    });
  });
}

cfWireSlide($('#cfTurns'), 'THINK_TURNS');
cfWireSlide($('#cfEmo'), 'EMOTION_SENSITIVITY');
cfWireSlide($('#cfEnergy'), 'ENERGY');

$$('#cfMode button').forEach(b => b.addEventListener('click', () => {
  S.settings.THINK_MODE = b.dataset.v;
  cfgPaint();
  cfPush('THINK_MODE', b.dataset.v);
}));

$$('#cfSpeed button').forEach(b => b.addEventListener('click', () => {
  const v = +b.dataset.v;
  S.settings.FLASH_LEVEL = v;
  cfgPaint();
  cfPush('FLASH_LEVEL', v);
}));

[['#cfPunct', 'USE_PUNCT_EMOJI'],
 ['#cfDeep', 'SHOW_DEEP_THINK'],
 ['#cfNet', 'FORCE_OFFLINE']].forEach(pair => {
  const el = $(pair[0]);
  if (!el) return;
  el.addEventListener('click', () => {
    const v = !S.settings[pair[1]];
    S.settings[pair[1]] = v;
    cfSwitch(el, v);
    cfCurLine();
    cfPush(pair[1], v);
  });
});

const cfResetBtn = $('#cfReset');
if (cfResetBtn) cfResetBtn.addEventListener('click', () => {
  Object.assign(S.settings, CF_DEF);
  Object.assign(S.think, { mode: CF_DEF.THINK_MODE, turns: CF_DEF.THINK_TURNS });
  paintThink();
  cfgPaint();
  const card = $('#cfgCard');
  if (card) card.classList.add('busy');
  api('/api/settings', CF_DEF).then(r => {
    if (card) card.classList.remove('busy');
    toast(r && r.ok ? '设置都回到默认了' : '没写进去，回头再试一次');
    return api('/api/settings').then(q => {
      if (q && q.ok && q.settings) { Object.assign(S.settings, q.settings); cfgPaint(); }
    });
  });
});

const cfgBtn = $('#cfgBtn');
if (cfgBtn) cfgBtn.addEventListener('click', () => { if (cfgIsOpen()) closeCfg(); else openCfg(); });

const cfgEl = $('#cfg');
if (cfgEl) cfgEl.addEventListener('pointerdown', ev => {
  if (ev.target === cfgEl) closeCfg();     /* 点卡片外头就收 */
});

const cfgCloseBtn = $('#cfgClose');
if (cfgCloseBtn) cfgCloseBtn.addEventListener('click', closeCfg);

/* ── 拖进来 / 挑一张 ───────────────────────────────────────────────── */
async function bgUpload(files) {
  const drop = $('#bgDrop');
  if (!drop || !files || !files.length) return;
  drop.classList.add('busy');
  let okN = 0, why = '';
  for (let i = 0; i < files.length; i++) {
    const fd = new FormData();
    /* 字段名固定叫 file —— 后端就认这一个 */
    fd.append('file', files[i], files[i].name);
    try {
      /* 这条得走裸 fetch：api() 会把 Content-Type 写成 JSON，multipart 就废了 */
      const r = await fetch('/api/background/upload', { method: 'POST', body: fd });
      const j = await r.json().catch(() => null);
      if (r.ok && j && j.ok !== false) okN++;
      else why = (j && j.err) || ('HTTP ' + r.status);
    } catch (_) { why = '没连上本地那个口子'; }
  }
  drop.classList.remove('busy');
  if (!okN) { toast(why || '没传上去'); return; }
  toast(okN === 1 ? '传上去一张' : '传上去 ' + okN + ' 张');
  await bgLoad(true);
}

const bgDrop = $('#bgDrop'), bgFile = $('#bgFile');
if (bgDrop && bgFile) {
  bgDrop.addEventListener('click', ev => { if (ev.target !== bgFile) bgFile.click(); });
  bgDrop.addEventListener('keydown', ev => {
    if (ev.key === 'Enter' || ev.key === ' ' || ev.key === 'Spacebar') {
      ev.preventDefault(); bgFile.click();
    }
  });
  bgFile.addEventListener('change', () => {
    const fs = Array.from(bgFile.files || []);
    bgFile.value = '';
    if (fs.length) bgUpload(fs);
  });

  ['dragenter', 'dragover'].forEach(t => bgDrop.addEventListener(t, ev => {
    ev.preventDefault();
    if (ev.dataTransfer) ev.dataTransfer.dropEffect = 'copy';
    bgDrop.classList.add('over');
  }));
  bgDrop.addEventListener('dragleave', ev => {
    /* 在块里头走来走去也会冒 dragleave，别把高亮抖掉 */
    if (ev.relatedTarget && bgDrop.contains(ev.relatedTarget)) return;
    bgDrop.classList.remove('over');
  });
  bgDrop.addEventListener('drop', ev => {
    ev.preventDefault();
    bgDrop.classList.remove('over');
    const fs = ev.dataTransfer ? Array.from(ev.dataTransfer.files || []) : [];
    if (!fs.length) return;
    const ok = fs.filter(f => /^image\/(png|jpeg|webp)$/i.test(f.type));
    if (!ok.length) { toast('只认 PNG / JPG / WEBP'); return; }
    if (ok.length < fs.length) toast('跳过了 ' + (fs.length - ok.length) + ' 个不是图的');
    bgUpload(ok);
  });
  /* 拖到面板别处松手，也别让浏览器把图当页面打开 */
  window.addEventListener('dragover', ev => { if (skinIsOpen()) ev.preventDefault(); });
  window.addEventListener('drop', ev => { if (skinIsOpen()) ev.preventDefault(); });
}

/* 上次选的那张，页面一进来就贴回去 —— 不等后端 */
bgRestore();

/* ── 键盘：Esc 收尾，Ctrl+K 开新对话 ───────────────────────────────── */
window.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    TW.skip = true; hidePop(); closeSheet(); closeSkin(); closeCfg(); artClose(); return;
  }
  if ((e.ctrlKey || e.metaKey) && (e.key === 'k' || e.key === 'K')) {
    e.preventDefault(); newChat();
  }
});

$('#bootRetry').addEventListener('click', () => location.reload());
window.addEventListener('resize', () => {
  paintThink();
  if (!pop.hidden) showPop();
  if (!tip.hidden) tip.hidden = true;
});

boot();
})();
