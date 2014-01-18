AC_INIT([graphMT], [2014-01-15], [avneesh@cs.cmu.edu])
AM_INIT_AUTOMAKE
AC_CONFIG_HEADERS(config.h)
AC_PROG_CC
AC_PROG_CXX
AC_LANG_CPLUSPLUS
AC_OPENMP
BOOST_REQUIRE([1.44])
BOOST_PROGRAM_OPTIONS

AC_ARG_WITH(eigen, [AC_HELP_STRING([--with-eigen=PATH], [(optional) path to Eigen linear algebra library])], 
		   [with_eigen=$withval],
		   [with_eigen=no]
		   )

if test "x$with_eigen" != 'xno'   
   SAVE_CPPFLAGS="$CPPFLAGS"
   CPPFLAGS="$CPPFLAGS -I${with_eigen}"

   AC_CHECK_HEADER(Eigen/Dense, [], [AC_MSG_ERROR([Cannot find Eigen in ${with_eigen}])])
then
   AC_CHECK_HEADER(Eigen/Dense, [], [AC_MSG_ERROR([Cannot find Eigen --with-eigen=/path/to/eigen])])
fi

AC_CONFIG_FILES([Makefile])

AC_OUTPUT
