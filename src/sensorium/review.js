const video = document.querySelector('video');
const items = [...document.querySelectorAll('.steps li')];
const now = document.querySelector('.now');
video.addEventListener('timeupdate', () => {
  const f = video.currentTime / video.duration || 0;
  document.querySelectorAll('.cursor').forEach(c => {
    const w = c.ownerSVGElement.viewBox.baseVal.width;
    c.setAttribute('x1', f * w); c.setAttribute('x2', f * w);
  });
  let cur = null;
  for (const li of items) { if (+li.dataset.t <= video.currentTime) cur = li; li.classList.remove('on'); }
  if (cur) { cur.classList.add('on'); now.textContent = cur.textContent.replace(/^\s*[\d.]+s\s*/, ''); }
});
items.forEach(li => li.addEventListener('click', () => { video.currentTime = +li.dataset.t; video.play(); }));
