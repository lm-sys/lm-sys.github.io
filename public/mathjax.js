window.MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
    },
    svg: {
      fontCache: 'global'
    },
    startup: {
      pageReady: () => {
        return MathJax.startup.defaultPageReady();
      }
    }
  };

